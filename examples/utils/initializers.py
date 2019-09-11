import csv
import gym
import json
import numpy as np
import os
import socket
import torch
import torch.optim as optim

from datetime import datetime
from torchcule.atari import Env as AtariEnv
from utils.openai.envs import create_vectorize_atari_env
from utils.runtime import cuda_device_str

try:
    import apex
    from apex.amp import __version__
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError('Please install apex from https://www.github.com/nvidia/apex to run this example.')

def args_initialize(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if (args.num_ales % args.world_size) != 0:
        raise ValueError('The num_ales({}) should be evenly divisible by the world_size({})'.format(args.num_ales, args.world_size))
    args.num_ales = int(args.num_ales / args.world_size)

    if ('batch_size' in args) and ((args.batch_size % args.world_size) != 0):
        raise ValueError('The batch_size({}) should be evenly divisible by the world_size({})'.format(args.batch_size, args.world_size))
    args.batch_size = int(args.num_ales / args.world_size)

    if args.distributed:
        args.seed += args.gpu
        torch.cuda.set_device(args.gpu)

        args.rank = int(os.environ['RANK']) if 'RANK' in os.environ else 0
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + args.gpu

        torch.distributed.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:8632',
                                             world_size=args.world_size, rank=args.rank)
    else:
        args.rank = 0

    if args.lr_scale:
        scaled_lr = args.lr * math.sqrt((args.num_ales * args.world_size) / 16)
        if args.rank == 0:
            print('Scaled learning rate from {:4.4f} to {:4.4f}'.format(args.lr, scaled_lr))
        args.lr = scaled_lr

    args.use_cuda_env = args.use_cuda_env and torch.cuda.is_available()
    args.cpu_train = args.cpu_train or (not torch.cuda.is_available())
    args.verbose = args.verbose and (args.rank == 0)

    np.random.seed(args.seed)
    torch.manual_seed(np.random.randint(1, 10000))
    if args.use_cuda_env or (not args.cpu_train):
        torch.cuda.manual_seed(np.random.randint(1, 10000))

    env_device = torch.device('cuda', args.gpu) if args.use_cuda_env else torch.device('cpu')
    train_device = torch.device('cuda', args.gpu) if (not args.cpu_train) else torch.device('cpu')

    return env_device, train_device

def env_initialize(args, device):
    if args.use_openai:
        train_env = create_vectorize_atari_env(args.env_name, args.seed, args.num_ales,
                                               episode_life=args.episodic_life,
                                               clip_rewards=args.clip_rewards,
                                               max_frames=args.max_episode_length)
        observation = torch.from_numpy(train_env.reset()).squeeze(1)
    else:
        train_env = AtariEnv(args.env_name, args.num_ales, color_mode='gray', repeat_prob=0.0,
                             device=device, rescale=True, episodic_life=args.episodic_life,
                             clip_rewards=args.clip_rewards, frameskip=4)
        train_env.train()
        observation = train_env.reset(initial_steps=args.ale_start_steps, verbose=args.verbose).squeeze(-1)

    if args.use_openai_test_env:
        test_env = create_vectorize_atari_env(args.env_name, args.seed, args.evaluation_episodes,
                                              episode_life=False, clip_rewards=False)
        test_env.reset()
    else:
        test_env = AtariEnv(args.env_name, args.evaluation_episodes, color_mode='gray', repeat_prob=0.0,
                            device='cpu', rescale=True, episodic_life=False, clip_rewards=False, frameskip=4)

    return train_env, test_env, observation

def log_initialize(args, device):
    if args.rank == 0:
        if args.output_filename:
            train_csv_file = open(args.output_filename, 'w', newline='')
            train_csv_writer = csv.writer(train_csv_file, delimiter=',')
            train_csv_writer.writerow(['frames','fps','total_time',
                                       'rmean','rmedian','rmin','rmax','rstd',
                                       'lmean','lmedian','lmin','lmax','lstd',
                                       'entropy','value_loss','policy_loss'])

            eval_output_filename = '.'.join([''.join(args.output_filename.split('.')[:-1] + ['_test']), 'csv'])
            eval_csv_file = open(eval_output_filename, 'w', newline='')
            eval_csv_file.write(json.dumps(vars(args)))
            eval_csv_file.write('\n')
            eval_csv_writer = csv.writer(eval_csv_file, delimiter=',')
            eval_csv_writer.writerow(['frames','total_time',
                                      'rmean','rmedian','rmin','rmax','rstd',
                                      'lmean','lmedian','lmin','lmax','lstd'])
        else:
            train_csv_file, train_csv_writer = None, None
            eval_csv_file, eval_csv_writer = None, None

        if args.plot:
            from tensorboardX import SummaryWriter
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            log_dir = os.path.join(args.log_dir, current_time + '_' + socket.gethostname())
            summary_writer = SummaryWriter(log_dir=log_dir)
            for k, v in vars(args).items():
                summary_writer.add_text(k, str(v))
        else:
            summary_writer = None
    else:
        train_csv_file, train_csv_writer, eval_csv_file, eval_csv_writer, summary_writer = None, None, None, None, None

    if args.verbose:
        print()
        print('PyTorch  : {}'.format(torch.__version__))
        print('CUDA     : {}'.format(torch.backends.cudnn.m.cuda))
        print('CUDNN    : {}'.format(torch.backends.cudnn.version()))
        print('APEX     : {}'.format('.'.join([str(i) for i in apex.amp.__version__.VERSION])))
        print('GYM      : {}'.format(gym.version.VERSION))
        print()

        if device.type == 'cuda':
            print(cuda_device_str(device.index), flush=True)

    return train_csv_file, train_csv_writer, eval_csv_file, eval_csv_writer, summary_writer

def model_initialize(args, model, device):
    model = model.to(device).train()

    if args.verbose:
        print(model)
        args.model_name = model.name()

    if args.use_adam:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, eps=args.eps, alpha=args.alpha)

    if device.type == 'cuda':
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.opt_level,
                                          loss_scale=args.loss_scale
                                         )

        if args.distributed:
            model = DDP(model, delay_allreduce=True)

    return model, optimizer

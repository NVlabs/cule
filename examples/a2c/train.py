import csv
import math
import json
import os
import torch
import time

import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from datetime import datetime
from tqdm import tqdm
from torchcule.atari import Env as AtariEnv
from utils.openai.envs import create_vectorize_atari_env
from utils.runtime import cuda_device_str

from model import ActorCritic
from helper import callback, evaluate

try:
    import apex
    from apex.amp import __version__
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError('Please install apex from https://www.github.com/nvidia/apex to run this example.')

def train(args, callback=callback):
	args.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
	args.distributed = (args.world_size > 1) or args.multiprocessing_distributed

	ngpus_per_node = torch.cuda.device_count() if args.num_gpus_per_node == -1 else args.num_gpus_per_node
	if args.multiprocessing_distributed:
		# Since we have ngpus_per_node processes per node, the total world_size
		# needs to be adjusted accordingly
		args.world_size = ngpus_per_node * args.world_size
		# Use torch.multiprocessing.spawn to launch distributed processes: the
		# main_worker process function
		torch.multiprocessing.spawn(worker, nprocs=ngpus_per_node, args=(ngpus_per_node, callback, args))
	else:
		# Simply call main_worker function
		worker(args.local_rank, ngpus_per_node, callback, args)

def worker(gpu, ngpus_per_node, callback, args):
    args.gpu = gpu

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
    args.no_cuda_train = (not args.no_cuda_train) and torch.cuda.is_available()
    args.verbose = args.verbose and (args.rank == 0)

    np.random.seed(args.seed)
    torch.manual_seed(np.random.randint(1, 10000))
    if args.use_cuda_env or (args.no_cuda_train == False):
        torch.cuda.manual_seed(np.random.randint(1, 10000))

    env_device = torch.device('cuda', args.gpu) if args.use_cuda_env else torch.device('cpu')
    train_device = torch.device('cuda', args.gpu) if (args.no_cuda_train == False) else torch.device('cpu')

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
            writer = SummaryWriter(log_dir=log_dir)
            for k, v in vars(args).items():
                writer.add_text(k, str(v))

        print()
        print('PyTorch  : {}'.format(torch.__version__))
        print('CUDA     : {}'.format(torch.backends.cudnn.m.cuda))
        print('CUDNN    : {}'.format(torch.backends.cudnn.version()))
        print('APEX     : {}'.format('.'.join([str(i) for i in apex.amp.__version__.VERSION])))
        print()

    if train_device.type == 'cuda':
        print(cuda_device_str(train_device.index), flush=True)

    if args.use_openai:
        train_env = create_vectorize_atari_env(args.env_name, args.seed, args.num_ales,
                                               episode_life=args.episodic_life, clip_rewards=False,
                                               max_frames=args.max_episode_length)
        observation = torch.from_numpy(train_env.reset()).squeeze(1)
    else:
        train_env = AtariEnv(args.env_name, args.num_ales, color_mode='gray', repeat_prob=0.0,
                             device=env_device, rescale=True, episodic_life=args.episodic_life,
                             clip_rewards=False, frameskip=4)
        train_env.train()
        observation = train_env.reset(initial_steps=args.ale_start_steps, verbose=args.verbose).squeeze(-1)

    if args.use_openai_test_env:
        test_env = create_vectorize_atari_env(args.env_name, args.seed, args.evaluation_episodes,
                                              episode_life=False, clip_rewards=False)
        test_env.reset()
    else:
        test_env = AtariEnv(args.env_name, args.evaluation_episodes, color_mode='gray', repeat_prob=0.0,
                            device='cpu', rescale=True, episodic_life=False, clip_rewards=False, frameskip=4)

    model = ActorCritic(args.num_stack, train_env.action_space, normalize=args.normalize, name=args.env_name)
    model = model.to(train_device).train()

    if args.rank == 0:
        print(model)
        args.model_name = model.name()

    if args.use_adam:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, eps=args.eps, alpha=args.alpha)

    model, optimizer = amp.initialize(model, optimizer,
                                      opt_level=args.opt_level,
                                      loss_scale=args.loss_scale
                                     )

    if args.distributed:
        model = DDP(model, delay_allreduce=True)

    num_frames_per_iter = args.num_ales * args.num_steps
    total_steps = math.ceil(args.t_max / (args.world_size * num_frames_per_iter))

    shape = (args.num_steps + 1, args.num_ales, args.num_stack, *train_env.observation_space.shape[-2:])
    states = torch.zeros(shape, device=train_device, dtype=torch.float32)
    states[0, :, -1] = observation.to(device=train_device, dtype=torch.float32)

    shape = (args.num_steps + 1, args.num_ales)
    values  = torch.zeros(shape, device=train_device, dtype=torch.float32)
    returns = torch.zeros(shape, device=train_device, dtype=torch.float32)

    shape = (args.num_steps, args.num_ales)
    rewards = torch.zeros(shape, device=train_device, dtype=torch.float32)
    masks = torch.zeros(shape, device=train_device, dtype=torch.float32)
    actions = torch.zeros(shape, device=train_device, dtype=torch.long)

    # These variables are used to compute average rewards for all processes.
    episode_rewards = torch.zeros(args.num_ales, device=train_device, dtype=torch.float32)
    final_rewards = torch.zeros(args.num_ales, device=train_device, dtype=torch.float32)
    episode_lengths = torch.zeros(args.num_ales, device=train_device, dtype=torch.float32)
    final_lengths = torch.zeros(args.num_ales, device=train_device, dtype=torch.float32)

    if args.use_gae:
        gae = torch.zeros(args.num_ales, device=train_device, dtype=torch.float32)

    maybe_npy = lambda a: a.numpy() if args.use_openai else a

    torch.cuda.synchronize()

    iterator = range(total_steps)
    if args.rank == 0:
        iterator = tqdm(iterator)
        total_time = 0
        evaluation_offset = 0

    for update in iterator:

        T = args.world_size * update * num_frames_per_iter
        if (args.rank == 0) and (T >= evaluation_offset):
            evaluation_offset += args.evaluation_interval
            eval_lengths, eval_rewards = evaluate(args, T, total_time, model, test_env, eval_csv_writer, eval_csv_file)

            if args.plot:
                writer.add_scalar('eval/rewards_mean', eval_rewards.mean().item(), T, walltime=total_time)
                writer.add_scalar('eval/lengths_mean', eval_lengths.mean().item(), T, walltime=total_time)

        start_time = time.time()

        with torch.no_grad():

            for step in range(args.num_steps):
                value, logit = model(states[step])

                # store values
                values[step] = value.squeeze(-1)

                # convert actions to numpy and perform next step
                probs_action = F.softmax(logit, dim=1).multinomial(1).to(env_device)
                observation, reward, done, info = train_env.step(maybe_npy(probs_action))

                if args.use_openai:
                    # convert back to pytorch tensors
                    observation = torch.from_numpy(observation)
                    reward = torch.from_numpy(reward)
                    done = torch.from_numpy(done.astype(np.uint8))
                else:
                    observation = observation.squeeze(-1).unsqueeze(1)

                # move back to training memory
                observation = observation.to(device=train_device)
                reward = reward.to(device=train_device, dtype=torch.float32)
                done = done.to(device=train_device, dtype=torch.bool)
                probs_action = probs_action.to(device=train_device, dtype=torch.long)

                not_done = 1.0 - done.float()

                # update rewards and actions
                actions[step].copy_(probs_action.view(-1))
                masks[step].copy_(not_done)
                rewards[step].copy_(reward.sign())

                # update next observations
                states[step + 1, :, :-1].copy_(states[step, :, 1:].clone())
                states[step + 1] *= not_done.view(-1, *[1] * (observation.dim() - 1))
                states[step + 1, :, -1].copy_(observation.view(-1, *states.size()[-2:]))

                # update episodic reward counters
                episode_rewards += reward
                final_rewards[done] = episode_rewards[done]
                episode_rewards *= not_done

                episode_lengths += not_done
                final_lengths[done] = episode_lengths[done]
                episode_lengths *= not_done

            returns[-1] = values[-1] = model(states[-1])[0].data.squeeze(-1)

            if args.use_gae:
                gae.zero_()
                for step in reversed(range(args.num_steps)):
                    delta = rewards[step] + (args.gamma * values[step + 1] * masks[step]) - values[step]
                    gae = delta + (args.gamma * args.tau * masks[step] * gae)
                    returns[step] = gae + values[step]
            else:
                for step in reversed(range(args.num_steps)):
                    returns[step] = rewards[step] + (args.gamma * returns[step + 1] * masks[step])

        value, logit = model(states[:-1].view(-1, *states.size()[-3:]))

        log_probs = F.log_softmax(logit, dim=1)
        probs = F.softmax(logit, dim=1)

        action_log_probs = log_probs.gather(1, actions.view(-1).unsqueeze(-1))
        dist_entropy = -(log_probs * probs).sum(-1).mean()

        advantages = returns[:-1].view(-1).unsqueeze(-1) - value

        value_loss = advantages.pow(2).mean()
        policy_loss = -(advantages.clone().detach() * action_log_probs).mean()

        loss = value_loss * args.value_loss_coef + policy_loss - dist_entropy * args.entropy_coef
        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
        optimizer.step()

        states[0].copy_(states[-1])

        torch.cuda.synchronize()

        if args.rank == 0:
            iter_time = time.time() - start_time
            total_time += iter_time

            if args.plot:
                writer.add_scalar('train/rewards_mean', final_rewards.mean().item(), T, walltime=total_time)
                writer.add_scalar('train/lengths_mean', final_lengths.mean().item(), T, walltime=total_time)
                writer.add_scalar('train/learning_rate', scheduler.get_lr()[0], T, walltime=total_time)
                writer.add_scalar('train/value_loss', value_loss, T, walltime=total_time)
                writer.add_scalar('train/policy_loss', policy_loss, T, walltime=total_time)
                writer.add_scalar('train/entropy', dist_entropy, T, walltime=total_time)

            progress_data = callback(args, model, T, iter_time, final_rewards, final_lengths,
                                     value_loss.item(), policy_loss.item(), dist_entropy.item(),
                                     train_csv_writer, train_csv_file)
            iterator.set_postfix_str(progress_data)

    if args.plot and (args.rank == 0):
        writer.close()

    if args.use_openai:
        train_env.close()
    if args.use_openai_test_env:
        test_env.close()

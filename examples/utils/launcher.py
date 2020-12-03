import argparse
import errno
import math
import logging
import os
import psutil
import re
import shutil
import sys
import time
import torch
import warnings

from pprint import pprint

if sys.version_info.major == 2:
    import ConfigParser as configparser
    from distutils.spawn import find_executable
else:
    from shutil import which as find_executable
    import configparser

def add_global_parser_options(parser):
    parser.add_argument('--ale-start-steps', type=int, default=400, help='Number of steps used to initialize ALEs (default: 400)')
    parser.add_argument('--alpha', type=float, default=0.99, help='RMSprop optimizer alpha (default: 0.99)')
    parser.add_argument('--clip-rewards', action='store_true', default=False, help='Clip rewards to {-1, 0, +1}')
    parser.add_argument('--cpu-train', action='store_true', default=False, help='Use CPU for training updates')
    parser.add_argument('--env-name', type=str, default='PongNoFrameskip-v4', help='Atari game name')
    parser.add_argument('--eps', type=float, default=1e-5, help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--episodic-life', action='store_true', default=False, help='use end of life as end of episode')
    parser.add_argument('--evaluation-interval', type=int, default=int(1e6), help='Number of frames between evaluations (default: 1,000,000)')
    parser.add_argument('--evaluation-episodes', type=int, default=10, help='Number of evaluation episodes to average over (default: 10)')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--gpu', type=int, default=None, help='GPU ID (default: None)')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--log-dir', type=str, default='runs', help='tensorboardX log directory (default: runs)')
    parser.add_argument('--lr', type=float, default=0.00065, help='learning rate (default: 0.00065)')
    parser.add_argument('--max-episode-length', type=int, default=18000, help='maximum length of an episode (default: 18,000)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help='max norm of gradients (default: 0.5)')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--normalize', action='store_true', default=False, help='Normalize and center input to network')
    parser.add_argument('--num-ales', type=int, default=16, help='number of environments (default: 16)')
    parser.add_argument('--num-gpus-per-node', type=int, default=-1, help='Number of GPUs per node (default: -1 [use all available])')
    parser.add_argument('--output-filename', type=str, default=None, help='Output filename')
    parser.add_argument('--plot', action='store_true', default=False, help='Enable plotting with bokeh')
    parser.add_argument('--profile', action='store_true', default=False, help='Enable Nsight Systems profiling')
    parser.add_argument('--save-interval', type=int, default=0, help='Interval to save model to file (default: 0)')
    parser.add_argument('--seed', type=int, default=int(time.time()), help='random seed (default: time())')
    parser.add_argument('--t-max', type=int, default=int(50e6), help='Number of training steps (default: 50,000,000)')
    parser.add_argument('--verbose', action='store_true', default=False, help='Enable verbose logging')
    parser.add_argument('--use-adam', action='store_true', default=False, help='use ADAM optimizer')
    parser.add_argument('--use-cuda-env', action='store_true', default=False, help='use CUDA for ALE updates')
    parser.add_argument('--use-openai', action='store_true', default=False, help='Use OpenAI Gym environment')
    parser.add_argument('--use-openai-test-env', action='store_true', default=False, help='Use OpenAI Gym test environment')

    return parser

def dispatch(args, worker):
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')

    args.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = (args.world_size > 1) or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count() if args.num_gpus_per_node == -1 else args.num_gpus_per_node
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        torch.multiprocessing.spawn(worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        worker(args.local_rank, ngpus_per_node, args)

def maybe_restart(args, worker):
    '''Restarts the current program, with file objects and descriptors
       cleanup
    '''

    env = os.environ
    argv = sys.argv

    if args.profile:
        argv = [l for l in argv if l != '--profile']

        cublas_lib = 'libToolsInjectionCuBLAS64_10_0.so'
        cudnn_lib = 'libToolsInjectionCuDNN64_7_2.so'
        nsight_lib  = 'libToolsInjectionProxy64.so'
        nsight_dir  = os.path.join(os.environ['HOME'], 'Downloads', 'NsightSystems', 'Target-x86_64', 'x86_64')
        nsight_path = os.path.join(nsight_dir, nsight_lib)
        cublas_path = os.path.join(nsight_dir, cublas_lib)
        cudnn_path = os.path.join(nsight_dir, cudnn_lib)

        if not os.path.isfile(nsight_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), nsight_path)

        extra_env = {
                'QUADD_INJECTION_PROXY': 'cuBLAS,cuDNN,CUDA,NVTX',
                'LD_PRELOAD': nsight_path
              }

        env.update(extra_env)

        executable = sys.executable

    if args.profile:
        try:
            p = psutil.Process(os.getpid())
            for handler in p.open_files() + p.connections():
                os.close(handler.fd)
        except Exception as e:
            logging.error(e)

        print('Executing...\n{}\n'.format(' '.join([executable] + argv)))
        argv += [env]
        os.execle(executable, executable, *argv)
    else:
        dispatch(args, worker)

# argparse initialization adapted from example at
# https://stackoverflow.com/questions/3609852/which-is-the-best-way-to-allow-configuration-options-be-overridden-at-the-comman
def main(add_extra_parser_options, worker):
    argv = sys.argv[1:]
    conf_parser = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            add_help=False)
    conf_parser.add_argument('-c', '--conf-file', help='Specify config file')
    args, remaining_argv = conf_parser.parse_known_args(args=argv)

    defaults = {}

    if args.conf_file:
        config = configparser.SafeConfigParser()
        config.read(args.conf_file)
        defaults.update(dict(config.items('Defaults')))
        for key, value in [(k,v.lower()) for k,v in defaults.items()]:
            if value in ('yes', 'true'):
                defaults[key] = True
            elif value in ('no', 'false'):
                defaults[key] = False
            elif value in ('none'):
                defaults[key] = None

    parser = argparse.ArgumentParser(description='CuLE', parents=[conf_parser])
    parser = add_global_parser_options(parser)
    parser = add_extra_parser_options(parser)
    parser.set_defaults(**defaults)
    args = parser.parse_args(remaining_argv)

    if args.local_rank == 0:
        pprint(vars(args))

    maybe_restart(args, worker)

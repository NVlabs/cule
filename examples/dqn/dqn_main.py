import argparse
import configparser
import errno
import logging
import os
import psutil
import re
import sys
import time

from pprint import pprint
from train import train

def add_parser_options(parser):
    parser.add_argument('--num-ales', type=int, default=32, help='number of environments (default: 32)')
    parser.add_argument('--env-name', type=str, default='PongNoFrameskip-v4', help='ATARI game (default: PongNoFrameskip-v4)')
    parser.add_argument('--t-max', type=int, default=int(50e6), metavar='STEPS', help='Number of training steps (default: 50,000,000)')
    parser.add_argument('--max-episode-length', type=int, default=int(18e3), metavar='LENGTH', help='Max episode length (18,000)')
    parser.add_argument('--history-length', type=int, default=4, help='Number of consecutive states processed')
    parser.add_argument('--hidden-size', type=int, default=512, help='Network hidden size')
    parser.add_argument('--noisy-std', type=float, default=0.1, metavar='σ', help='Initial standard deviation of noisy linear layers')
    parser.add_argument('--atoms', type=int, default=51, help='Discretised size of value distribution')
    parser.add_argument('--v-min', type=float, default=-10, help='Minimum of value distribution support')
    parser.add_argument('--v-max', type=float, default=10, help='Maximum of value distribution support')
    parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
    parser.add_argument('--memory-capacity', type=int, default=int(10e5), metavar='CAPACITY', help='Experience replay memory capacity (default: 1,000,000)')
    parser.add_argument('--replay-frequency', type=int, default=4, metavar='k', help='The number of the gradient step updates (intensity) per step in the environment')
    parser.add_argument('--priority-exponent', type=float, default=0.7, metavar='ω', help='Prioritised experience replay exponent (originally denoted α)')
    parser.add_argument('--priority-weight', type=float, default=0.5, metavar='β', help='Initial prioritised experience replay importance sampling weight')
    parser.add_argument('--multi-step', type=int, default=3, metavar='n', help='Number of steps for multi-step return (default: 3)')
    parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor (default: 0.99)')
    parser.add_argument('--max-grad-norm', type=float, default=1.00, help='max norm of gradients (default: 1.00)')
    parser.add_argument('--target-update', type=int, default=int(32000), metavar='τ', help='Number of frames after which to update target network (default: 32,000)')
    parser.add_argument('--reward-clip', action='store_true', default=False, help='Clip rewards to {-1, 0, +1}')
    parser.add_argument('--lr', type=float, default=6.25e-05, metavar='η', help='Learning rate')
    parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--learn-start', type=int, default=int(40e3), help='Number of steps before starting training (default: 40,000)')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
    parser.add_argument('--evaluation-interval', type=int, default=int(1e6), help='Number of frames between evaluations (default: 1,000,000)')
    parser.add_argument('--evaluation-episodes', type=int, default=10, help='Number of evaluation episodes to average over (default: 10)')
    parser.add_argument('--evaluation-size', type=int, default=500, help='Number of transitions to use for validating Q')
    parser.add_argument('--log-interval', type=int, default=int(10e4), help='Number of training steps between logging status (default: 100,000)')
    parser.add_argument('--seed', type=int, default=int(time.time()), help='Random seed')
    parser.add_argument('--use-openai', action='store_true', help='Use OpenAI environment')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID (default: 0)')
    parser.add_argument('--num-gpus-per-node', type=int, default=-1, help='Number of GPUs per node (default: -1 [use all available])')
    parser.add_argument('--epsilon-start', type=float, default=1.0, help='Start epsilon (default: 1.0)')
    parser.add_argument('--epsilon-final', type=float, default=0.1, help='Final epsilon (default: 0.1')
    parser.add_argument('--epsilon-frames', type=float, default=int(15e6), help='Epsilon decay frames (default: 15,000,000')
    parser.add_argument('--use-cuda-env', action='store_true', default=False, help='use CUDA for ALE updates')
    parser.add_argument('--normalize', action='store_true', default=False, help='Normalize and center input to network')
    parser.add_argument('--priority-replay', action='store_true', default=False, help='Enable prioritized experience replay')
    parser.add_argument('--categorical', action='store_true', default=False, help='Enable distributional RL model')
    parser.add_argument('--double-q', action='store_true', default=False, help='Enable double Q learning')
    parser.add_argument('--dueling', action='store_true', default=False, help='Enable dueling DQN model')
    parser.add_argument('--noisy-linear', action='store_true', default=False, help='Enable noisy linear layers')
    parser.add_argument('--rainbow', action='store_true', default=False, help='Enable all components for Rainbow DQN')
    parser.add_argument('--ale-start-steps', type=int, default=int(4000), help='Number of ALE warmup steps')
    parser.add_argument('--opt-level', type=str, default='O0')
    parser.add_argument('--loss-scale', type=str, default=None)
    parser.add_argument('--output-filename', type=str, default=None, help='Output filename')
    parser.add_argument('--profile', action='store_true', default=False, help='Enable Nsight Systems profiling')
    parser.add_argument('--verbose', action='store_true', default=False, help='Enable verbose logging')
    parser.add_argument('--log-dir', default='runs', type=str, help='tensorboardX log directory (default: runs)')
    parser.add_argument('--plot', action='store_true', default=False, help='Enable plotting with tensorboardX')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    return parser

def maybe_restart(args):
    """Restarts the current program, with file objects and descriptors
       cleanup
    """

    env = os.environ
    argv = sys.argv

    if args.profile:
        cublas_lib = 'libToolsInjectionCuBLAS64_10_0.so'
        cudnn_lib = 'libToolsInjectionCuDNN64_7_3.so'
        nsight_lib  = 'libToolsInjectionProxy64.so'
        nsight_dir  = os.path.join(os.environ['HOME'], 'Downloads', 'NsightSystems', 'Target-x86_64', 'x86_64')
        nsight_path = os.path.join(nsight_dir, nsight_lib)
        cublas_path = os.path.join(nsight_dir, cublas_lib)
        cudnn_path = os.path.join(nsight_dir, cudnn_lib)

        if not os.path.isfile(nsight_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), nsight_path)

        extra_env = {
                'QUADD_INJECTION_PROXY': 'cuBLAS,cuDNN,CUDA,NVTX',
                'LD_PRELOAD': ' '.join([nsight_path, cudnn_path, cublas_path])
              }

        env.update(extra_env)

        executable = sys.executable
        argv = [l for l in argv if l != '--profile']

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
        train(args)

# argparse initialization adapted from example at
# https://stackoverflow.com/questions/3609852/which-is-the-best-way-to-allow-configuration-options-be-overridden-at-the-comman
def dqn_main(argv=sys.argv[1:]):
    conf_parser = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            add_help=False)
    conf_parser.add_argument('-c', '--conf-file', metavar='FILE', help='Specify config file')
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

    parser = argparse.ArgumentParser(description='DQN', parents=[conf_parser])
    parser = add_parser_options(parser)
    parser.set_defaults(**defaults)
    args = parser.parse_args(remaining_argv)

    if args.rainbow:
        args.categorical = True
        args.double_q = True
        args.dueling = True
        args.noisy_linear = True
        args.priority_replay = True

    if not args.profile and (args.local_rank == 0):
        pprint(vars(args))

    maybe_restart(args)

if __name__ == "__main__":
    dqn_main()

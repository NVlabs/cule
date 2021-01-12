import os
import sys

_path = os.path.abspath(os.path.pardir)
if not _path in sys.path:
    sys.path = [_path] + sys.path

from utils.launcher import main

from train import worker

def bfs_parser_options(parser):
    parser.add_argument('--entropy-coef', type=float, default=0.01, help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--lr-scale', action='store_true', default=False, help='Scale the learning rate with the batch-size')
    parser.add_argument('--num-stack', type=int, default=4, help='number of images in a stack (default: 4)')
    parser.add_argument('--num-steps', type=int, default=5, help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--max-depth', type=int, default=4, help='depth of action space for BFS traversal (default: 4)')
    parser.add_argument('--tau', type=float, default=1.00, help='parameter for GAE (default: 1.00)')
    parser.add_argument('--use-gae', action='store_true', default=False, help='use generalized advantage estimation')
    parser.add_argument('--value-loss-coef', type=float, default=0.5, help='value loss coefficient (default: 0.5)')

    return parser

def bfs_main():
    if sys.version_info.major == 3:
        from train import worker
    else:
        worker = None

    sys.exit(main(bfs_parser_options, worker))

if __name__ == '__main__':
    bfs_main()

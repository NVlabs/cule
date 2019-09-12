import os
import sys

current_path = os.path.dirname(os.path.realpath(__file__))
_path = os.path.abspath(os.path.join(current_path, os.pardir))
if not _path in sys.path:
    sys.path = [_path] + sys.path

from a2c.a2c_main import a2c_parser_options
from utils.launcher import main

def ppo_parser_options(parser):
    parser = a2c_parser_options(parser)

    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--clip-epsilon', type=float, default=0.1, help='ppo clip parameter (default: 0.1)')
    parser.add_argument('--ppo-epoch', type=int, default=3, help='Number of ppo epochs (default: 3)')

    return parser

def ppo_main():
    if sys.version_info.major == 3:
        from train import worker
    else:
        worker = None

    sys.exit(main(ppo_parser_options, worker))

if __name__ == '__main__':
    ppo_main()

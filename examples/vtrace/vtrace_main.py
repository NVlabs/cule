import os
import sys

_path = os.path.abspath(os.path.pardir)
if not _path in sys.path:
    sys.path = [_path] + sys.path

from a2c.a2c_main import a2c_parser_options
from utils.launcher import main

def vtrace_parser_options(parser):
    parser = a2c_parser_options(parser)

    parser.add_argument('--c-hat', type=int, default=1.0, help='Trace cutting truncation level (default: 1.0)')
    parser.add_argument('--rho-hat', type=int, default=1.0, help='Temporal difference truncation level (default: 1.0)')
    parser.add_argument('--num-minibatches', type=int, default=16, help='number of mini-batches in the set of environments (default: 16)')
    parser.add_argument('--num-steps-per-update', type=int, default=1, help='number of steps per update (default: 1)')

    return parser

if __name__ == '__main__':
    if sys.version_info.major == 3:
        from train import train
    else:
        train = None

    sys.exit(main(vtrace_parser_options, train))

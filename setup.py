import argparse
import os
import sys

from distutils.cmd import Command
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

try:
    from examples.utils.runtime import get_device_props
    codes = sorted(set([str(p.major) + str(p.minor) for p in get_device_props()]))
except:
    print('Warning: Could not find nvcc in path. Compiling with sm_70 by default.')
    codes = ['70']

arch_gencode = ['-arch=sm_' + codes[0]] + ['-gencode=arch=compute_{0},code=sm_{0}'.format(code) for code in codes]
base_dir = os.path.dirname(os.path.realpath(__file__))

sources = [os.path.join('torchcule', f) for f in ['frontend.cpp', 'backend.cu']]
include_dirs = [base_dir, os.path.join(base_dir, 'third_party', 'agency')]
libraries = ['gomp', 'z']
cxx_flags = []
nvcc_flags = arch_gencode + ['-O3', '-Xptxas=-v', '-Xcompiler=-Wall,-Wextra,-fpermissive']

parser = argparse.ArgumentParser('CuLE', add_help=False)
parser.add_argument('--fastbuild', action='store_true', default=False, help='Build CuLE supporting only 2K roms')
parser.add_argument('--compiler', type=str, default='gcc', help='Host compiler (default: gcc)')
args, remaining_argv = parser.parse_known_args()
sys.argv = ['setup.py'] + remaining_argv

if args.fastbuild:
    nvcc_flags += ['-DCULE_FAST_COMPILE']

nvcc_flags += ['-ccbin={}'.format(args.compiler)]

setup(name='torchcule',
      version='0.1.0',
      description='A GPU RL environment package for PyTorch',
      url='https://github.com/NVlabs/cule',
      author='Steven Dalton',
      author_email='sdalton1@gmail.com',
      install_requires=['gym>=0.9.5'],
      ext_modules=[
          CUDAExtension('torchcule_atari',
              sources=sources,
              include_dirs=include_dirs,
              libraries=libraries,
              extra_compile_args={'cxx': cxx_flags, 'nvcc': nvcc_flags}
              )
          ],
      # Exclude the build files.
      packages=find_packages(exclude=['build']),
      cmdclass={
          'build_ext': BuildExtension,
          }
      )

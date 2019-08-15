import argparse
import os
import sys

from Cython.Distutils import build_ext

from distutils.cmd import Command
from setuptools import find_packages, setup, Extension
from examples.utils.runtime import Runtime

try:
    from examples.utils.runtime import get_device_props
    codes = sorted(set([str(p.major) + str(p.minor) for p in get_device_props()]))
except:
    print('Warning: Could not find nvcc in path. Compiling with sm_70 by default.')
    codes = ['70']

arch_gencode = ['-arch=sm_' + codes[0]] + ['-gencode=arch=compute_{0},code=sm_{0}'.format(code) for code in codes]
base_dir = os.path.dirname(os.path.realpath(__file__))

runtime = Runtime()
CUDA = runtime._locate()

sources = [os.path.join('torchcule', f) for f in ['frontend.cpp', 'backend.cu']]
third_party_dir = os.path.join(base_dir, 'third_party')
include_dirs = [base_dir, os.path.join(third_party_dir, 'agency'), os.path.join(third_party_dir, 'pybind11', 'include'), CUDA['include']]
libraries = ['gomp', 'z']
cxx_flags = []
nvcc_flags = arch_gencode + ['-O3', '-Xptxas=-v', '-Xcompiler=-Wall,-Wextra,-fpermissive,-fPIC']

parser = argparse.ArgumentParser('CuLE', add_help=False)
parser.add_argument('--fastbuild', action='store_true', default=False, help='Build CuLE supporting only 2K roms')
parser.add_argument('--compiler', type=str, default='gcc', help='Host compiler (default: gcc)')
args, remaining_argv = parser.parse_known_args()
sys.argv = ['setup.py'] + remaining_argv

if args.fastbuild:
    nvcc_flags += ['-DCULE_FAST_COMPILE']

nvcc_flags += ['-ccbin={}'.format(args.compiler)]

def customize_compiler_for_nvcc(self):
    """Inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.
    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on.
    """

    # Tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # Save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # Now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1
            # translated from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # Reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # Inject our redefined _compile method into the class
    self._compile = _compile

# Run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)

setup(name='torchcule',
      version='0.1.0',
      description='A GPU RL environment package for PyTorch',
      url='https://github.com/NVlabs/cule',
      author='Steven Dalton',
      author_email='sdalton1@gmail.com',
      install_requires=['gym>=0.9.5'],
      ext_modules=[
          Extension('torchcule_atari',
              sources=sources,
              include_dirs=include_dirs,
              libraries=libraries,
              extra_compile_args={'gcc': cxx_flags, 'nvcc': nvcc_flags}
              )
          ],
      # Exclude the build files.
      packages=find_packages(exclude=['build']),
      cmdclass={
          'build_ext': custom_build_ext,
          }
      )

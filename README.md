[![Build Status](https://travis-ci.com/NVlabs/cule.svg?branch=master)](https://travis-ci.com/NVlabs/cule)

![ALT](/media/images/System.png "Deep RL System Overview")

# CuLE 0.1.0

_CuLE 0.1.0 - July 2019_

CuLE is a CUDA port of the Atari Learning Environment (ALE) and is
designed to accelerate the development and evaluation of deep
reinforcement algorithms using Atari games. Our CUDA Learning
Environment (CuLE) overcomes many limitations of existing CPU- based
Atari emulators and scales naturally to multi-GPU systems.  It leverages
the parallelization capability of GPUs to run thousands of Atari games
simultaneously; by rendering frames directly on the GPU, CuLE avoids the
bottleneck arising from the limited CPU-GPU communication bandwidth.

# Compatibility

CuLE performs best when compiled with the [CUDA 10.0 Toolkit](https://developer.nvidia.com/cuda-toolkit).
It is currently incompatible with CUDA 10.1.

We have tested the following environments.

|**Operating System** | **Compiler** |
|-----------------|----------|
| Ubuntu 16.04 | GCC 5.4.0 |
| Ubuntu 18.04 | GCC 7.3.0 |

CuLE runs successfully on the following NVIDIA GPUs, and it is expected to be efficient on
any Maxwell-, Pascal-, Volta-, and Turing-architecture NVIDIA GPUs.

|**GPU**|
|---|
|NVIDIA Tesla P100|
|NVIDIA Tesla V100|
|NVIDIA TitanV|

# Building CuLE

```
$ git clone --recursive https://github.com/NVlabs/cule
$ python setup.py install
```

# Project Structure

```
cule/
  cule/
  env/
  examples/
  media/
  third_party/
  torchcule/
```

Several example programs are also distributed with the CuLE library. They are
contained in the following directories.

```
examples/
  a2c/
  dqn/
  ppo/
  vtrace/
  utils/
  visualize/
```

# Citing

```
@misc{dalton2019gpuaccelerated,
   title={GPU-Accelerated Atari Emulation for Reinforcement Learning},
   author={Steven Dalton and Iuri Frosio and Michael Garland},
   year={2019},
   eprint={1907.08467},
   archivePrefix={arXiv},
   primaryClass={cs.LG}
}
```

# About

CuLE is released by NVIDIA Corporation as Open Source software under the
3-clause "New" BSD license.

# Copyright

Copyright (c) 2017-2019, NVIDIA CORPORATION.  All rights reserved.

```
  Redistribution and use in source and binary forms, with or without modification, are permitted
  provided that the following conditions are met:
      * Redistributions of source code must retain the above copyright notice, this list of
        conditions and the following disclaimer.
      * Redistributions in binary form must reproduce the above copyright notice, this list of
        conditions and the following disclaimer in the documentation and/or other materials
        provided with the distribution.
      * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
        to endorse or promote products derived from this software without specific prior written
        permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
  IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
  FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
  OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
  STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

#pragma once

#include <cuda.h>
#include <sstream>
#include <stdexcept>

namespace cule
{
namespace cuda
{

void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

} // end namespace cuda
} // end namespace cule


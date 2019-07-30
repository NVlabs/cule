#pragma once

#include <sstream>
#include <stdexcept>

// macro with optional string parameter
// taken from http://stackoverflow.com/questions/3046889/optional-parameters-with-c-macros
#define CULE_ASSERT_0()    CULE_ASSERT_1(true)
#define CULE_ASSERT_1(A)   CULE_ASSERT_2(A, std::string())
#define CULE_ASSERT_2(A,B) CULE_ASSERT_3(A, B, std::runtime_error)
#define CULE_ASSERT_3(A,B,C)                                      \
if(not (A))                                                         \
{                                                                   \
    std::ostringstream m;                                           \
    m << __FILE__ << ":" << __LINE__ << " in " << __func__ << "\n"; \
    m << B << "\n";                                                 \
    throw C(m.str());                                               \
}

// The interim macro that simply strips the excess and ends up with the required macro
#define CULE_ASSERT_X(x,A,B,C,FUNC,...) FUNC

// The macro that the programmer uses
#ifdef __CUDA_ARCH__
#define CULE_ASSERT(...)
#else
#define CULE_ASSERT(...) CULE_ASSERT_X(,##__VA_ARGS__,  \
                           CULE_ASSERT_3(__VA_ARGS__),    \
                           CULE_ASSERT_2(__VA_ARGS__),    \
                           CULE_ASSERT_1(__VA_ARGS__),    \
                           CULE_ASSERT_0(__VA_ARGS__)     \
                          )
#endif

// Macro for unimplemented functions
#define CULE_NOT_IMPLEMENTED CULE_ASSERT(false, std::string(__PRETTY_FUNCTION__));

// CULE_RETURNS() is used to avoid writing boilerplate "->decltype(x) { return x; }" phrases.
// see https://gist.github.com/dabrahams/1457531 for details
#define CULE_RETURNS(...) -> decltype(__VA_ARGS__) { return (__VA_ARGS__); } typedef int CULE_RETURNS_CAT(CULE_RETURNS_, __LINE__)
#define CULE_RETURNS_CAT_0(x, y) x ## y
#define CULE_RETURNS_CAT(x, y) CULE_RETURNS_CAT_0(x,y)

#ifdef __CUDACC__
#define CULE_ERRCHK(ans) { cule::cuda::gpuAssert((ans), __FILE__, __LINE__); }

#define CULE_CUDA_PEEK_AND_SYNC     \
  CULE_ERRCHK(cudaPeekAtLastError());   \
  CULE_ERRCHK(cudaDeviceSynchronize());

#define CULE_ANNOTATION __host__ __device__ __forceinline__
#else
#define CULE_ANNOTATION inline
#define CULE_ERRCHK(ans)
#endif

#ifdef __CUDA_ARCH__
#define CULE_ARRAY_ACCESSOR(name) cule::atari::gpu_##name
#else
#define CULE_ARRAY_ACCESSOR(name) cule::atari::name
#endif


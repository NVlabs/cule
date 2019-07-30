#pragma once

#include <stdint.h>

#define PCG_DEFAULT_MULTIPLIER_32 747796405U
#define PCG_DEFAULT_INCREMENT_32 2891336453U

#ifndef CURAND_2POW32_INV
#define CURAND_2POW32_INV (2.3283064e-10f)
#endif

namespace cule
{
namespace atari
{

class prng
{
  public:

    CULE_ANNOTATION
    prng(uint32_t& state)
      : state(state)
    {}

    CULE_ANNOTATION
    void initialize(const uint32_t seed)
    {
        state = 0U;
        sample();
        state += seed;
        sample();
    }

    CULE_ANNOTATION
    uint32_t sample()
    {
        uint32_t oldstate = state;

        // Advance internal state
        state = oldstate * PCG_DEFAULT_MULTIPLIER_32 + PCG_DEFAULT_INCREMENT_32;

        uint32_t word = ((oldstate >> ((oldstate >> 28U) + 4U)) ^ oldstate) * 277803737U;

        return (word >> 22U) ^ word;
    }

    CULE_ANNOTATION
    float sample_float()
    {
        return sample() * CURAND_2POW32_INV + (CURAND_2POW32_INV / 2.0f);
    }

  private:

    uint32_t& state;
};

} // end namespace atari
} // end namespace cule


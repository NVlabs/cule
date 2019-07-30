#pragma once

#include <cule/config.hpp>

namespace cule
{
namespace atari
{

// forward declaration
template <typename T, int bits> class bit_field;
template <typename T, typename ET, int bits> class flag_set;

} // end namespace atari
} // end namespace cule

#include <cule/atari/types/bitfield.hpp>
#include <cule/atari/types/flagset.hpp>

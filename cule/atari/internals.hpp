#pragma once

#include <cule/config.hpp>

#include <cule/atari/flags.hpp>
#include <cule/atari/types/types.hpp>

namespace cule
{
namespace atari
{

// internal type definitions
#ifdef FAST_TYPE
typedef unsigned _addr16_t;
typedef unsigned _addr15_t;
typedef unsigned _addr14_t;
typedef unsigned _addr8_t;
typedef unsigned _reg8_t;
typedef unsigned _alutemp_t;
typedef unsigned byte_t;
typedef unsigned word_t;
typedef unsigned uint_t;
#else // FAST_TYPE
typedef uint16_t _addr16_t;
typedef uint16_t _addr15_t;
typedef uint16_t _addr14_t;
typedef uint8_t  _addr8_t;
typedef uint8_t  _reg8_t;
typedef uint16_t _alutemp_t;
typedef uint8_t  byte_t;
typedef uint16_t word_t;
typedef uint32_t uint_t;
#endif // EXACT_TYPE

// address
typedef bit_field<_addr16_t,16> maddr_t;
typedef bit_field<_addr15_t,15> scroll_t, addr15_t;
typedef bit_field<_addr14_t,14> vaddr_t, addr14_t;
typedef bit_field<_addr8_t,8> maddr8_t, saddr_t;

// cpu
typedef uint8_t opcode_t;
typedef uint8_t operand_t;

// alu
typedef bit_field<uint8_t,8> operandb_t;
typedef bit_field<word_t,16> operandw_t;
typedef bit_field<_alutemp_t,8> alu_t;

// color
typedef uint32_t rgb32_t;
typedef uint16_t rgb16_t, rgb15_t;
typedef bit_field<_reg8_t, 8> reg_bit_field_t;

// others
typedef bit_field<unsigned,3> offset3_t;
typedef bit_field<unsigned,10> offset10_t;

using sys_flag_t = flag_set<uint32_t, SYS_FLAGS, 32>;
using tia_flag_t = flag_set<uint32_t, TIA_FLAGS, 28>;
using collision_t = flag_set<uint16_t, CollisionBit, 15>;

} // end namespace atari
} // end namespace cule


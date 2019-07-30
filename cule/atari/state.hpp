#pragma once

#include <cule/config.hpp>
#include <cule/macros.hpp>

#include <cule/atari/flags.hpp>
#include <cule/atari/internals.hpp>

#include <cstdlib>

namespace cule
{
namespace atari
{

struct state
{
    // m6502 vars
    _reg8_t		A; // accumulator
    _reg8_t		X, Y; // index

    maddr8_t SP; // stack pointer
    maddr_t	PC; // program counter
    maddr_t	addr; // effective address

    uint8_t value; // operand
    uint8_t noise;

    uint16_t cpuCycles;
    uint16_t bank;

    // controller vars
    int32_t resistance;

    // TIA vars
    uint32_t GRP;
    uint32_t HM;
    uint32_t PF;
    uint32_t POS;
    uint8_t CurrentGRP0;
    uint8_t CurrentGRP1;

    uint16_t collision;
    int16_t clockWhenFrameStarted;
    int32_t clockAtLastUpdate;
    int32_t dumpDisabledCycle;
    int32_t VSYNCFinishClock;
    int32_t lastHMOVEClock; // Color clock when last HMOVE occurred

    // m6532 vars
    uint32_t riotData;
    int32_t cyclesWhenTimerSet;
    int32_t cyclesWhenInterruptReset;

    // state flags
    sys_flag_t sysFlags;
    tia_flag_t tiaFlags;

    // frame data
    uint32_t frameData;
    uint32_t rand;
    int32_t score;

    // pointers
    uint32_t * ram;
    const uint8_t * rom;
    uint32_t * tia_update_buffer;

    uint32_t* CurrentPFMask;
    uint8_t * CurrentP0Mask;
    uint8_t * CurrentP1Mask;
    uint8_t * CurrentM0Mask;
    uint8_t * CurrentM1Mask;
    uint8_t * CurrentBLMask;
};

} // end namespace atari
} // end namespace cule


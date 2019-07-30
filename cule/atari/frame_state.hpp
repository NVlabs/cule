#pragma once

#include <cule/config.hpp>

#include <cule/atari/internals.hpp>

namespace cule
{
namespace atari
{

struct frame_state
{
    uint32_t Color;
    uint32_t GRP;
    uint32_t HM;
    uint32_t PF;
    uint32_t POS;
    uint8_t CurrentGRP0;
    uint8_t CurrentGRP1;

    int32_t clockWhenFrameStarted;
    int32_t clockAtLastUpdate;

    // Color clock when last HMOVE occured
    int32_t lastHMOVEClock;

    uint8_t playfieldPriorityAndScore;
    uint16_t cpuCycles;

    tia_flag_t tiaFlags;

    uint8_t* framePointer;
    const uint32_t* srcBuffer;

    uint32_t* CurrentPFMask;
    uint8_t * CurrentP0Mask;
    uint8_t * CurrentP1Mask;
    uint8_t * CurrentM0Mask;
    uint8_t * CurrentM1Mask;
    uint8_t * CurrentBLMask;
};

} // end namespace atari
} // end namespace cule


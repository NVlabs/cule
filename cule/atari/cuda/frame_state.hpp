#pragma once

#include <cule/config.hpp>

#include <cule/atari/frame_state.hpp>

namespace cule
{
namespace atari
{

CULE_ANNOTATION
void state_store_load_helper(frame_state& t, const frame_state& s)
{
    t.Color = s.Color;
    t.GRP = s.GRP;
    t.HM = s.HM;
    t.PF = s.PF;
    t.POS = s.POS;
    t.CurrentGRP0 = s.CurrentGRP0;
    t.CurrentGRP1 = s.CurrentGRP1;

    t.clockWhenFrameStarted = s.clockWhenFrameStarted;
    t.clockAtLastUpdate = s.clockAtLastUpdate;
    t.lastHMOVEClock = s.lastHMOVEClock;

    t.playfieldPriorityAndScore = s.playfieldPriorityAndScore;

    t.tiaFlags = s.tiaFlags;

    t.CurrentPFMask = s.CurrentPFMask;
    t.CurrentP0Mask = s.CurrentP0Mask;
    t.CurrentP1Mask = s.CurrentP1Mask;
    t.CurrentM0Mask = s.CurrentM0Mask;
    t.CurrentM1Mask = s.CurrentM1Mask;
    t.CurrentBLMask = s.CurrentBLMask;
}

} // end namespace atari
} // end namespace cule


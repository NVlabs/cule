#pragma once

#include <cule/config.hpp>
#include <cule/atari/state.hpp>

namespace cule
{
namespace atari
{

template<typename State>
CULE_ANNOTATION
void state_store_load_helper(State& t, const State& s)
{
    t.A = s.A;
    t.X = s.X;
    t.Y = s.Y;
    t.SP = s.SP;
    t.PC = s.PC;
    t.addr = s.addr;
    t.value = s.value;
    t.noise = s.noise;

    t.cpuCycles = s.cpuCycles;
    t.bank = s.bank;

    t.resistance = s.resistance;

    t.GRP = s.GRP;
    t.HM = s.HM;
    t.PF = s.PF;
    t.POS = s.POS;
    t.CurrentGRP0 = s.CurrentGRP0;
    t.CurrentGRP1 = s.CurrentGRP1;

    t.collision = s.collision;
    t.clockWhenFrameStarted = s.clockWhenFrameStarted;
    t.clockAtLastUpdate = s.clockAtLastUpdate;
    t.dumpDisabledCycle = s.dumpDisabledCycle;
    t.VSYNCFinishClock = s.VSYNCFinishClock;
    t.lastHMOVEClock = s.lastHMOVEClock;

    t.riotData = s.riotData;
    t.cyclesWhenTimerSet = s.cyclesWhenTimerSet;
    t.cyclesWhenInterruptReset = s.cyclesWhenInterruptReset;

    t.sysFlags = s.sysFlags;
    t.tiaFlags = s.tiaFlags;

    t.frameData = s.frameData;
    // t.rand = s.rand;
    t.score = s.score;

    t.CurrentPFMask = s.CurrentPFMask;
    t.CurrentP0Mask = s.CurrentP0Mask;
    t.CurrentP1Mask = s.CurrentP1Mask;
    t.CurrentM0Mask = s.CurrentM0Mask;
    t.CurrentM1Mask = s.CurrentM1Mask;
    t.CurrentBLMask = s.CurrentBLMask;
}

} // end namespace atari
} // end namespace cule


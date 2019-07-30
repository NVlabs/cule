#pragma once

#include <cule/config.hpp>
#include <cule/atari/controller.hpp>
#include <cule/atari/state.hpp>

namespace cule
{
namespace atari
{
namespace games
{
namespace bowling
{

template<typename State>
CULE_ANNOTATION
void reset(State& s)
{
    s.m_reward   = 0;
    s.m_score    = 0;
    s.tiaFlags.clear(FLAG_ALE_TERMINAL);
}

template<typename State>
CULE_ANNOTATION
void step(State& s)
{
    // update the reward
    s.m_reward = s.m_score - getScore(s);

    // handle KO
    setTerminal(s);
}

CULE_ANNOTATION
bool isMinimal(const Action &a)
{
    switch (a)
    {
    case ACTION_NOOP:
    case ACTION_FIRE:
    case ACTION_UP:
    case ACTION_DOWN:
    case ACTION_UPFIRE:
    case ACTION_DOWNFIRE:
        return true;
    default:
        return false;
    }
}

template<typename State>
CULE_ANNOTATION
int32_t lives(State&)
{
    return 0;
}

template<typename State>
CULE_ANNOTATION
void setTerminal(State& s)
{
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>(cule::atari::ram::read(s.ram, 0xA4) > 0x10);
}

template<typename State>
CULE_ANNOTATION
int32_t score(State& s)
{
    return cule::atari::games::getDecimalScore(s, 0xA1, 0xA6);
}

template<typename State>
CULE_ANNOTATION
int32_t reward(State& s)
{
    return score(s) - s.score;
}

} // end namespace bowling
} // end namespace games
} // end namespace atari
} // end namespace cule


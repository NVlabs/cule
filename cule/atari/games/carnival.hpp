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
namespace carnival
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
    using cule::atari::games::getDecimalScore;
    using cule::atari::ram::read;

    // update the reward
    int score = getDecimalScore(s, 0xAE, 0xAD);
    score *= 10;
    s.m_reward = score - s.m_score;
    s.m_score = score;

    // update terminal status
    int ammo = ram::read(s, 0x83);
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>(ammo < 1);
}

CULE_ANNOTATION
bool isMinimal(const Action &a)
{
    switch (a)
    {
    case ACTION_NOOP:
    case ACTION_FIRE:
    case ACTION_RIGHT:
    case ACTION_LEFT:
    case ACTION_RIGHTFIRE:
    case ACTION_LEFTFIRE:
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
    // update terminal status
    int ammo = cule::atari::ram::read(s.ram, 0x83);
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>(ammo < 1);
}

template<typename State>
CULE_ANNOTATION
int32_t score(State& s)
{
    return 10 * cule::atari::games::getDecimalScore(s, 0xAE, 0xAD);
}

template<typename State>
CULE_ANNOTATION
int32_t reward(State& s)
{
    return score(s) - s.score;
}

} // end namespace carnival
} // end namespace games
} // end namespace atari
} // end namespace cule


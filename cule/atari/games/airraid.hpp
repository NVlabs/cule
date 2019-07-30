#pragma once

#include <cule/config.hpp>
#include <cule/atari/actions.hpp>
#include <cule/atari/controller.hpp>
#include <cule/atari/state.hpp>

namespace cule
{
namespace atari
{
namespace games
{
namespace airraid
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
    int score = getDecimalScore(s, 0xAA, 0xA9, 0xA8);
    s.m_reward = score - s.m_score;
    s.m_score = score;

    // update terminal status
    int lives = ram::read(s, 0xA7);
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>(lives == 0xFF);
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
int32_t lives(State& s)
{
    return cule::atari::ram::read(s.ram, 0xA7);
}

template<typename State>
CULE_ANNOTATION
void setTerminal(State& s)
{
    // update terminal status
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>(lives(s) == 0xFF);
}

template<typename State>
CULE_ANNOTATION
int32_t score(State& s)
{
    return cule::atari::games::getDecimalScore(s, 0xAA, 0xA9, 0xA8);
}

template<typename State>
CULE_ANNOTATION
int32_t reward(State& s)
{
    return score(s) - s.score;
}

} // end namespace airraid
} // end namespace games
} // end namespace atari
} // end namespace cule


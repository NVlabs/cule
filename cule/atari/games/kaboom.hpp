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
namespace kaboom
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
    int score = getDecimalScore(s, 0xA5, 0xA4, 0xA3);
    s.m_reward = score - s.m_score;
    s.m_score = score;

    // update terminal status
    int lives = ram::read(s, 0xA1);
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>((lives == 0x0) || (s.m_score == 999999));
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
        return true;
    default:
        return false;
    }
}

template<typename State>
CULE_ANNOTATION
int32_t lives(State& s)
{
    return cule::atari::ram::read(s.ram, 0xA1);
}

template<typename State>
CULE_ANNOTATION
 void setTerminal(State& s)
{
    int32_t score = cule::atari::games::getDecimalScore(s, 0xA5, 0xA4, 0xA3);
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>((lives(s) == 0x0) || (score == 999999));
}

template<typename State>
CULE_ANNOTATION
int32_t score(State& s)
{
    return cule::atari::games::getDecimalScore(s, 0xA5, 0xA4, 0xA3);
}

template<typename State>
CULE_ANNOTATION
int32_t reward(State& s)
{
    return score(s) - s.score;
}

} // end namespace kaboom
} // end namespace games
} // end namespace atari
} // end namespace cule


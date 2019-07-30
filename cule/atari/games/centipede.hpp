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
namespace centipede
{

template<typename State>
CULE_ANNOTATION
void reset(State& s)
{
    s.m_reward   = 0;
    s.m_score    = 0;
    s.tiaFlags.clear(FLAG_ALE_TERMINAL);
    s.m_lives		 = 3;
}

template<typename State>
CULE_ANNOTATION
void step(State& s)
{
    using cule::atari::games::getDecimalScore;
    using cule::atari::ram::read;

    // update the reward
    int score = getDecimalScore(s, 118, 117, 116);
    s.m_reward = score - s.m_score;
    s.m_score = score;

    // HACK: the score sometimes gets reset before termination; ignoring for now.
    if (s.m_reward < 0) s.m_reward = 0.0;

    // Maximum of 8 lives
    s.m_lives = ((ram::read(s, 0xED) >> 4) & 0x7) + 1;

    // update terminal status
    int some_bit = ram::read(s, 0xA6) & 0x40;
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>(some_bit != 0);
}

CULE_ANNOTATION
bool isMinimal(const Action &a)
{
    switch (a)
    {
    case ACTION_NOOP:
    case ACTION_FIRE:
    case ACTION_UP:
    case ACTION_RIGHT:
    case ACTION_LEFT:
    case ACTION_DOWN:
    case ACTION_UPRIGHT:
    case ACTION_UPLEFT:
    case ACTION_DOWNRIGHT:
    case ACTION_DOWNLEFT:
    case ACTION_UPFIRE:
    case ACTION_RIGHTFIRE:
    case ACTION_LEFTFIRE:
    case ACTION_DOWNFIRE:
    case ACTION_UPRIGHTFIRE:
    case ACTION_UPLEFTFIRE:
    case ACTION_DOWNRIGHTFIRE:
    case ACTION_DOWNLEFTFIRE:
        return true;
    default:
        return false;
    }
}

template<typename State>
CULE_ANNOTATION
int32_t lives(State& s)
{
    // Maximum of 8 lives
    return ((cule::atari::ram::read(s.ram, 0xED) >> 4) & 0x7) + 1;
}

template<typename State>
CULE_ANNOTATION
void setTerminal(State& s)
{
    // update terminal status
    int some_bit = cule::atari::ram::read(s.ram, 0xA6) & 0x40;
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>(some_bit != 0);
}

template<typename State>
CULE_ANNOTATION
int32_t score(State& s)
{
    return cule::atari::games::getDecimalScore(s, 118, 117, 116);
}

template<typename State>
CULE_ANNOTATION
int32_t reward(State& s)
{
    int32_t m_reward = score(s) - s.score;

    // HACK: the score sometimes gets reset before termination; ignoring for now.
    if (m_reward < 0)
      m_reward = 0;

    return m_reward;
}

} // end namespace centipede
} // end namespace games
} // end namespace atari
} // end namespace cule


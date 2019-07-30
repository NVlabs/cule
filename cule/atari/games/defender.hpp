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
namespace defender
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
    int mult = 1, score = 0;
    for (int digit = 0; digit < 6; digit++)
    {
        int v = ram::read(s, 0x9C + digit) & 0xF;
        // A indicates a 0 which we don't display
        if (v == 0xA) v = 0;
        score += v * mult;
        mult *= 10;
    }
    s.m_reward = score - s.m_score;
    s.m_score = score;

    // update terminal status
    s.m_lives = ram::read(s, 0xC2);
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>(s.m_lives == 0);
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
    return cule::atari::ram::read(s.ram, 0xC2);
}

template<typename State>
CULE_ANNOTATION
void setTerminal(State& s)
{
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>(lives(s) == 0);
}

template<typename State>
CULE_ANNOTATION
int32_t score(State& s)
{
    int mult = 1, m_score = 0;
    for (int digit = 0; digit < 6; digit++)
    {
        int v = cule::atari::ram::read(s.ram, 0x9C + digit) & 0xF;
        // a indicates a 0 which we don't display
        if (v == 0xA) v = 0;
        m_score += v * mult;
        mult *= 10;
    }

    return m_score;
}

template<typename State>
CULE_ANNOTATION
int32_t reward(State& s)
{
    return score(s) - s.score;
}

} // end namespace defender
} // end namespace games
} // end namespace atari
} // end namespace cule


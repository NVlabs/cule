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
namespace asteroids
{

template<typename State>
CULE_ANNOTATION
void reset(State& s)
{
    s.m_reward   = 0;
    s.m_score    = 0;
    s.tiaFlags.clear(FLAG_ALE_TERMINAL);
    s.m_lives    = 4;
}

template<typename State>
CULE_ANNOTATION
void step(State& s)
{
    using cule::atari::games::getDecimalScore;
    using cule::atari::ram::read;

    // update the reward
    int score = getDecimalScore(s, 0xBE, 0xBD);
    score *= 10;
    s.m_reward = score - s.m_score;

    // Deal with score wrapping. In truth this should be done for all games and in a more
    // uniform fashion.
    if (s.m_reward < 0)
    {
        const int WRAP_SCORE = 100000;
        s.m_reward += WRAP_SCORE;
    }
    s.m_score = score;

    // update terminal status
    int byte = ram::read(s, 0xBC);
    s.m_lives = (byte - (byte & 15)) >> 4;
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
    case ACTION_UPFIRE:
    case ACTION_RIGHTFIRE:
    case ACTION_LEFTFIRE:
    case ACTION_DOWNFIRE:
    case ACTION_UPRIGHTFIRE:
    case ACTION_UPLEFTFIRE:
        return true;
    default:
        return false;
    }
}

template<typename State>
CULE_ANNOTATION
int32_t lives(State& s)
{
    // update terminal status
    int byte = cule::atari::ram::read(s.ram, 0xBC);
    return (byte - (byte & 0xF)) >> 4;
}

template<typename State>
CULE_ANNOTATION
void setTerminal(State& s)
{
    // update terminal status
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>(lives(s) == 0);
}

template<typename State>
CULE_ANNOTATION
int32_t score(State& s)
{
    return 10 * cule::atari::games::getDecimalScore(s, 0xBE, 0xBD);
}

template<typename State>
CULE_ANNOTATION
int32_t reward(State& s)
{
    int32_t m_reward = score(s) - s.score;
    /* s.score = m_score; */

    // Deal with score wrapping. In truth this should be done for all games and in a more
    // uniform fashion.
    if (m_reward < 0)
    {
        const int WRAP_SCORE = 100000;
        m_reward += WRAP_SCORE;
    }

    return m_reward;
}

} // end namespace asteroids
} // end namespace games
} // end namespace atari
} // end namespace cule


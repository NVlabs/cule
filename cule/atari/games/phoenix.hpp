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
namespace phoenix
{

template<typename State>
CULE_ANNOTATION
 void reset(State& s)
{
    s.m_reward   = 0;
    s.m_score    = 0;
    s.tiaFlags.clear(FLAG_ALE_TERMINAL);
    s.m_lives 	 = 5;
}

template<typename State>
CULE_ANNOTATION
 void step(State& s)
{
    using cule::atari::games::getDecimalScore;
    using cule::atari::ram::read;

    // update the reward
    int score = getDecimalScore(s, 0xC8, 0xC9) * 10;
    score += ram::read(s, 0xC7) >> 4;
    score *= 10;
    int reward = score - s.m_score;
    s.m_reward = reward;
    s.m_score = score;

    // update terminal status
    int state_byte = ram::read(s, 0xCC);
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>(state_byte == 0x80);
    // Technically seems to only go up to 5
    s.m_lives = ram::read(s, 0xCB) & 0x7;
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
    case ACTION_DOWN:
    case ACTION_RIGHTFIRE:
    case ACTION_LEFTFIRE:
    case ACTION_DOWNFIRE:
        return true;
    default:
        return false;
    }
}

template<typename State>
CULE_ANNOTATION
int32_t lives(State& s)
{
    // Technically seems to only go up to 5
    return cule::atari::ram::read(s.ram, 0xCB) & 0x7;
}

template<typename State>
CULE_ANNOTATION
 void setTerminal(State& s)
{
    // update terminal status
    int32_t state_byte = cule::atari::ram::read(s.ram, 0xCC);
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>(state_byte == 0x80);
}

template<typename State>
CULE_ANNOTATION
int32_t score(State& s)
{
    // update the reward
    int m_score = cule::atari::games::getDecimalScore(s, 0xC8, 0xC9) * 10;
    m_score += cule::atari::ram::read(s.ram, 0xC7) >> 4;
    m_score *= 10;

    return m_score;
}

template<typename State>
CULE_ANNOTATION
int32_t reward(State& s)
{
    return score(s) - s.score;
}

} // end namespace phoenix
} // end namespace games
} // end namespace atari
} // end namespace cule


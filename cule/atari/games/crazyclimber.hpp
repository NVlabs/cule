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
namespace crazyclimber
{

template<typename State>
CULE_ANNOTATION
void reset(State& s)
{
    s.m_reward   = 0;
    s.m_score    = 0;
    s.tiaFlags.clear(FLAG_ALE_TERMINAL);
    s.m_lives		 = 5;
}

template<typename State>
CULE_ANNOTATION
void step(State& s)
{
    using cule::atari::games::getDecimalScore;
    using cule::atari::ram::read;

    // update the reward
    int score = 0;
    int digit = ram::read(s, 0x82);
    score += digit;
    digit = ram::read(s, 0x83);
    score += 10 * digit;
    digit = ram::read(s, 0x84);
    score += 100 * digit;
    digit = ram::read(s, 0x85);
    score += 1000 * digit;
    score *= 100;
    s.m_reward = score - s.m_score;
    if (s.m_reward < 0) s.m_reward = 0;
    s.m_score = score;

    // update terminal status
    s.m_lives = ram::read(s, 0xAA);
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>(s.m_lives == 0);
}

CULE_ANNOTATION
bool isMinimal(const Action &a)
{
    switch (a)
    {
    case ACTION_NOOP:
    case ACTION_UP:
    case ACTION_RIGHT:
    case ACTION_LEFT:
    case ACTION_DOWN:
    case ACTION_UPRIGHT:
    case ACTION_UPLEFT:
    case ACTION_DOWNRIGHT:
    case ACTION_DOWNLEFT:
        return true;
    default:
        return false;
    }
}

template<typename State>
CULE_ANNOTATION
int32_t lives(State& s)
{
    return cule::atari::ram::read(s.ram, 0xAA);
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
    int32_t m_score = 0;
    int32_t digit = cule::atari::ram::read(s.ram, 0x82);
    m_score += digit;
    digit = cule::atari::ram::read(s.ram, 0x83);
    m_score += 10 * digit;
    digit = cule::atari::ram::read(s.ram, 0x84);
    m_score += 100 * digit;
    digit = cule::atari::ram::read(s.ram, 0x85);
    m_score += 1000 * digit;
    m_score *= 100;

    return m_score;
}

template<typename State>
CULE_ANNOTATION
int32_t reward(State& s)
{
    int32_t m_reward = score(s) - s.score;

    if(m_reward < 0)
      m_reward = 0;

    return m_reward;
}

} // end namespace crazyclimber
} // end namespace games
} // end namespace atari
} // end namespace cule


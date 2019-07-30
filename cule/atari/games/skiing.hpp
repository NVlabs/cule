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
namespace skiing
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
    int centiseconds = getDecimalScore(s, 0xEA, 0xE9);
    int minutes = ram::read(s, 0xE8);
    int score = minutes * 6000 + centiseconds;
    int reward = s.m_score - score; // negative reward for time
    s.m_reward = reward;
    s.m_score = score;

    // update terminal status
    int end_flag = ram::read(s, 0x91);
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>(end_flag == 0xFF);
}

CULE_ANNOTATION
bool isMinimal(const Action &a)
{
    switch (a)
    {
    case ACTION_NOOP:
    case ACTION_RIGHT:
    case ACTION_LEFT:
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
    int end_flag = cule::atari::ram::read(s.ram, 0x91);
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>(end_flag == 0xFF);
}

template<typename State>
CULE_ANNOTATION
int32_t score(State& s)
{
    // update the reward
    int centiseconds = cule::atari::games::getDecimalScore(s, 0xEA, 0xE9);
    int minutes = cule::atari::ram::read(s.ram, 0xE8);
    return minutes * 6000 + centiseconds;
}

template<typename State>
CULE_ANNOTATION
int32_t reward(State& s)
{
    return s.score - score(s);
}

} // end namespace skiing
} // end namespace games
} // end namespace atari
} // end namespace cule


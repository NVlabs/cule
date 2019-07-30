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
namespace freeway
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
    int score = getDecimalScore(s, 103, -1);
    int reward = score - s.m_score;
    if (reward < 0) reward = 0;
    if (reward > 1) reward = 1;
    s.m_reward = reward;
    s.m_score = score;

    // update terminal status
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>(ram::read(s, 22) == 1);
}

CULE_ANNOTATION
bool isMinimal(const Action &a)
{
    switch (a)
    {
    case ACTION_NOOP:
    case ACTION_UP:
    case ACTION_DOWN:
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
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>(cule::atari::ram::read(s.ram, 22) == 1);
}

template<typename State>
CULE_ANNOTATION
int32_t score(State& s)
{
    return cule::atari::games::getDecimalScore(s, 103, -1);
}

template<typename State>
CULE_ANNOTATION
int32_t reward(State& s)
{
    int32_t m_reward = score(s) - s.score;

    if (m_reward < 0) m_reward = 0;
    if (m_reward > 1) m_reward = 1;

    return m_reward;
}

} // end namespace freeway
} // end namespace games
} // end namespace atari
} // end namespace cule


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
namespace journeyescape
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
    int score = getDecimalScore(s, 0x92, 0x91, 0x90);
    int reward = score - s.m_score;
    if (reward == 50000) reward = 0;  // HACK: ignoring starting cash
    s.m_reward = reward;
    s.m_score = score;

    // update terminal status
    int minutes = ram::read(s, 0x95);
    int seconds = ram::read(s, 0x96);
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>((minutes == 0) && (seconds == 0));
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
int32_t lives(State&)
{
    return 0;
}

template<typename State>
CULE_ANNOTATION
void setTerminal(State& s)
{
    // update terminal status
    int minutes = cule::atari::ram::read(s.ram, 0x95);
    int seconds = cule::atari::ram::read(s.ram, 0x96);
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>((minutes == 0) && (seconds == 0));
}

template<typename State>
CULE_ANNOTATION
int32_t score(State& s)
{
    return cule::atari::games::getDecimalScore(s, 0x92, 0x91, 0x90);
}

template<typename State>
CULE_ANNOTATION
int32_t reward(State& s)
{
    int32_t m_score = score(s);
    int32_t m_reward = m_score - s.score;

    if (m_reward == 50000) m_reward = 0;  // HACK: ignoring starting cash

    return m_reward;
}

} // end namespace journeyescape
} // end namespace games
} // end namespace atari
} // end namespace cule


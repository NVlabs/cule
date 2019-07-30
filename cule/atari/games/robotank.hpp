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
namespace robotank
{

template<typename State>
CULE_ANNOTATION
void reset(State& s)
{
    s.m_reward   = 0;
    s.m_score    = 0;
    s.tiaFlags.clear(FLAG_ALE_TERMINAL);
    s.m_lives 	 = 4;
}

template<typename State>
CULE_ANNOTATION
void step(State& s)
{
    using cule::atari::games::getDecimalScore;
    using cule::atari::ram::read;

    // update the reward
    int dead_squadrons = ram::read(s, 0xB6);
    int dead_tanks = ram::read(s, 0xB5);
    int score = dead_squadrons * 12 + dead_tanks;
    int reward = score - s.m_score;
    s.m_reward = reward;
    s.m_score = score;

    // update terminal status
    int termination_flag = ram::read(s, 0xB4);
    int lives = ram::read(s, 0xA8);
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>((lives == 0) && (termination_flag == 0xFF));

    s.m_lives = (lives & 0xF) + 1;
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
    int lives = cule::atari::ram::read(s.ram, 0xA8);
    return (lives & 0xF) + 1;
}

template<typename State>
CULE_ANNOTATION
 void setTerminal(State& s)
{
    // update terminal status
    int termination_flag = cule::atari::ram::read(s.ram, 0xB4);
    int lives = cule::atari::ram::read(s.ram, 0xA8);
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>((lives == 0) && (termination_flag == 0xFF));
}

template<typename State>
CULE_ANNOTATION
int32_t score(State& s)
{
    // update the reward
    int dead_squadrons = cule::atari::ram::read(s.ram, 0xB6);
    int dead_tanks = cule::atari::ram::read(s.ram, 0xB5);
    return dead_squadrons * 12 + dead_tanks;
}

template<typename State>
CULE_ANNOTATION
int32_t reward(State& s)
{
    return score(s) - s.score;
}

} // end namespace robotank
} // end namespace games
} // end namespace atari
} // end namespace cule


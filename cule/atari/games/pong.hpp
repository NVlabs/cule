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
namespace pong
{

template<typename State>
CULE_ANNOTATION
 void reset(State& s)
{
    s.m_reward   = 0;
    s.m_score    = 0;
    s.m_lives    = 0;
    s.tiaFlags.clear(FLAG_ALE_TERMINAL);
}

template<typename State>
CULE_ANNOTATION
 void step(State& s)
{
    // update the reward
    int x = cule::atari::ram::read(s, 13); // cpu score
    int y = cule::atari::ram::read(s, 14); // player score
    int score = y - x;
    s.m_reward = score - s.m_score;
    s.m_score = score;

    // update terminal status
    // (game over when a player reaches 21)
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>(x == 21 || y == 21);
}

CULE_ANNOTATION
 bool isMinimal(const Action& a)
{
    switch (a)
    {
    case ACTION_NOOP:
    case ACTION_FIRE:
    case ACTION_RIGHT:
    case ACTION_LEFT:
    case ACTION_RIGHTFIRE:
    case ACTION_LEFTFIRE:
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
    // update the reward
    const int32_t x = cule::atari::ram::read(s.ram, 13); // cpu score
    const int32_t y = cule::atari::ram::read(s.ram, 14); // player score

    // update terminal status
    // (game over when a player reaches 21)
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>(x == 21 || y == 21);
}

template<typename State>
CULE_ANNOTATION
int32_t score(State& s)
{
    // update the reward
    int32_t x = cule::atari::ram::read(s.ram, 13); // cpu score
    int32_t y = cule::atari::ram::read(s.ram, 14); // player score

    return (y - x);
}

template<typename State>
CULE_ANNOTATION
int32_t reward(State& s)
{
    return score(s) - s.score;
}

} // end namespace pong
} // end namespace games
} // end namespace atari
} // end namespace cule


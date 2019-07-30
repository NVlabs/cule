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
namespace spaceinvaders
{

template<typename State>
CULE_ANNOTATION
void reset(State& s)
{
    s.m_reward   = 0;
    s.m_score    = 0;
    s.m_lives    = 3;
    s.tiaFlags.clear(FLAG_ALE_TERMINAL);
}

template<typename State>
CULE_ANNOTATION
void step(State& s)
{
    using cule::atari::games::getDecimalScore;
    using cule::atari::ram::read;

    // update the reward
    int score = getDecimalScore(s, 0xE8, 0xE6);

    // reward cannot get negative in this game. When it does, it means that the score has looped
    // (overflow)
    s.m_reward = score - s.m_score;
    if(s.m_reward < 0)
    {
        // 10000 is the highest possible score
        const int maximumScore = 10000;
        s.m_reward = (maximumScore - s.m_score) + score;
    }
    s.m_score = score;
    s.m_lives = ram::read(s, 0xC9);

    // update terminal status
    // If bit 0x80 is on, then game is over
    int some_byte = ram::read(s, 0x98);
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>((some_byte & 0x80) || (s.m_lives == 0));
}

CULE_ANNOTATION
bool isMinimal(const Action &a)
{
    switch (a)
    {
    case ACTION_NOOP:
    case ACTION_LEFT:
    case ACTION_RIGHT:
    case ACTION_FIRE:
    case ACTION_LEFTFIRE:
    case ACTION_RIGHTFIRE:
        return true;
    default:
        return false;
    }
}

template<typename State>
CULE_ANNOTATION
int32_t lives(State& s)
{
    return cule::atari::ram::read(s.ram, 0xC9);
}

template<typename State>
CULE_ANNOTATION
void setTerminal(State& s)
{
    // update terminal status
    // If bit 0x80 is on, then game is over
    int some_byte = cule::atari::ram::read(s.ram, 0x98);
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>((some_byte & 0x80) || (lives(s) == 0));
}

template<typename State>
CULE_ANNOTATION
int32_t score(State& s)
{
    return cule::atari::games::getDecimalScore(s, 0xE8, 0xE6);
}

template<typename State>
CULE_ANNOTATION
int32_t reward(State& s)
{
    return score(s) - s.score;
}

} // end namespace spaceinvaders
} // end namespace games
} // end namespace atari
} // end namespace cule


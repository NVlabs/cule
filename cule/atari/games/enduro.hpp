#pragma once

#include <cule/config.hpp>
#include <cule/atari/actions.hpp>
#include <cule/atari/controller.hpp>
#include <cule/atari/state.hpp>

#include <cassert>

namespace cule
{
namespace atari
{
namespace games
{
namespace enduro
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
    int score = 0;
    int level = ram::read(s, 0xAD);
    if (level != 0)
    {
        int cars_passed = getDecimalScore(s, 0xAB, 0xAC);
        if (level == 1) cars_passed = 200 - cars_passed;
        else if (level >= 2) cars_passed = 300 - cars_passed;
        else assert(false);

        // First level has 200 cars
        if (level >= 2)
        {
            score = 200;
            // For every level after the first, 300 cars
            score += (level - 2) * 300;
        }
        score += cars_passed;
    }
    s.m_reward = score - s.m_score;
    s.m_score = score;

    // update terminal status
    //int timeLeft = ram::read(&system, 0xB1);
    int deathFlag = ram::read(s, 0xAF);
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>(deathFlag == 0xFF);
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
    case ACTION_DOWNRIGHT:
    case ACTION_DOWNLEFT:
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
    // update terminal status
    int deathFlag = cule::atari::ram::read(s.ram, 0xAF);
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>(deathFlag == 0xFF);
}

template<typename State>
CULE_ANNOTATION
int32_t score(State& s)
{
    int m_score = 0;
    int level = cule::atari::ram::read(s.ram, 0xAD);
    if (level != 0)
    {
        int cars_passed = cule::atari::games::getDecimalScore(s, 0xAB, 0xAC);
        if (level == 1) cars_passed = 200 - cars_passed;
        else if (level >= 2) cars_passed = 300 - cars_passed;
        else assert(false);

        // First level has 200 cars
        if (level >= 2)
        {
            m_score = 200;
            // For every level after the first, 300 cars
            m_score += (level - 2) * 300;
        }
        m_score += cars_passed;
    }

    return m_score;
}

template<typename State>
CULE_ANNOTATION
int32_t reward(State& s)
{
   return score(s) - s.score;
}

} // end namespace enduro
} // end namespace games
} // end namespace atari
} // end namespace cule


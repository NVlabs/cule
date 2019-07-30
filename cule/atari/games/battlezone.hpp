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
namespace battlezone
{

template<typename State>
CULE_ANNOTATION
void reset(State& s)
{
    s.m_reward   = 0;
    s.m_score    = 0;
    s.tiaFlags.clear(FLAG_ALE_TERMINAL);
    s.m_lives    = 5;
}

template<typename State>
CULE_ANNOTATION
void step(State& s)
{
    using cule::atari::games::getDecimalScore;
    using cule::atari::ram::read;

    // update the reward
    int first_val = ram::read(s, 0x9D);
    int first_right_digit = first_val & 15;
    int first_left_digit = (first_val - first_right_digit) >> 4;
    if (first_left_digit == 10) first_left_digit = 0;

    int second_val = ram::read(s, 0x9E);
    int second_right_digit = second_val & 15;
    int second_left_digit = (second_val - second_right_digit) >> 4;
    if (second_right_digit == 10) second_right_digit = 0;
    if (second_left_digit == 10) second_left_digit = 0;

    int score = 0;
    score += first_left_digit;
    score += 10 * second_right_digit;
    score += 100 * second_left_digit;
    score *= 1000;
    s.m_reward = score - s.m_score;
    s.m_score = score;

    // update terminal status
    s.m_lives = ram::read(s, 0xBA) & 0xF;
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
    return cule::atari::ram::read(s.ram, 0xBA) & 0xF;
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
    // update the reward
    int first_val = cule::atari::ram::read(s.ram, 0x9D);
    int first_right_digit = first_val & 15;
    int first_left_digit = (first_val - first_right_digit) >> 4;
    if (first_left_digit == 10) first_left_digit = 0;

    int second_val = cule::atari::ram::read(s.ram, 0x9E);
    int second_right_digit = second_val & 15;
    int second_left_digit = (second_val - second_right_digit) >> 4;
    if (second_right_digit == 10) second_right_digit = 0;
    if (second_left_digit == 10) second_left_digit = 0;

    int m_score = 0;
    m_score += first_left_digit;
    m_score += 10 * second_right_digit;
    m_score += 100 * second_left_digit;
    m_score *= 1000;

    return m_score;
}

template<typename State>
CULE_ANNOTATION
int32_t reward(State& s)
{
    return score(s) - s.score;
}

} // end namespace battlezone
} // end namespace games
} // end namespace atari
} // end namespace cule


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
namespace stargunner
{

template<typename State>
CULE_ANNOTATION
void reset(State& s)
{
    s.m_reward   = 0;
    s.m_score    = 0;
    s.tiaFlags.clear(FLAG_ALE_TERMINAL);
    s.m_lives 	 = 5;
    s.tiaFlags.set(FLAG_ALE_STARTED);
}

template<typename State>
CULE_ANNOTATION
void step(State& s)
{
    using cule::atari::games::getDecimalScore;
    using cule::atari::ram::read;

    // update the reward
    int lower_digit = ram::read(s, 0x83) & 0x0F;
    if (lower_digit == 10) lower_digit = 0;
    int middle_digit = ram::read(s, 0x84) & 0x0F;
    if (middle_digit == 10) middle_digit = 0;
    int higher_digit = ram::read(s, 0x85) & 0x0F;
    if (higher_digit == 10) higher_digit = 0;
    int digit_4 = ram::read(s, 0x86) & 0x0F;
    if (digit_4 == 10) digit_4 = 0;
    int score = lower_digit + 10 * middle_digit + 100 * higher_digit + 1000 * digit_4;
    score *= 100;
    int reward = score - s.m_score;
    s.m_reward = reward;
    s.m_score = score;

    // update terminal status
    int lives_byte = ram::read(s, 0x87);
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>(lives_byte == 0);

    // We record when the game starts, which is needed to deal with the lives == 6 starting
    // situation
    s.tiaFlags.template change<FLAG_ALE_STARTED>(lives_byte == 0x05);

    s.m_lives = s.tiaFlags[FLAG_ALE_STARTED] ? (lives_byte & 0xF) : 5;
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
    // update terminal status
    int lives_byte = cule::atari::ram::read(s.ram, 0x87);
    return s.tiaFlags[FLAG_ALE_STARTED] ? (lives_byte & 0xF) : 5;
}

template<typename State>
CULE_ANNOTATION
void setTerminal(State& s)
{
    // update terminal status
    int lives_byte = cule::atari::ram::read(s.ram, 0x87);
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>(lives_byte == 0);

    // We record when the game starts, which is needed to deal with the lives == 6 starting
    // situation
    s.tiaFlags.template change<FLAG_ALE_STARTED>(lives_byte == 0x05);
}

template<typename State>
CULE_ANNOTATION
int32_t score(State& s)
{
    // update the reward
    int lower_digit = cule::atari::ram::read(s.ram, 0x83) & 0x0F;
    if (lower_digit == 10) lower_digit = 0;
    int middle_digit = cule::atari::ram::read(s.ram, 0x84) & 0x0F;
    if (middle_digit == 10) middle_digit = 0;
    int higher_digit = cule::atari::ram::read(s.ram, 0x85) & 0x0F;
    if (higher_digit == 10) higher_digit = 0;
    int digit_4 = cule::atari::ram::read(s.ram, 0x86) & 0x0F;
    if (digit_4 == 10) digit_4 = 0;
    int m_score = lower_digit + 10 * middle_digit + 100 * higher_digit + 1000 * digit_4;
    m_score *= 100;

    return m_score;
}

template<typename State>
CULE_ANNOTATION
int32_t reward(State& s)
{
    return score(s) - s.score;
}

} // end namespace stargunner
} // end namespace games
} // end namespace atari
} // end namespace cule


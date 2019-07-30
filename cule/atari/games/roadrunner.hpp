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
namespace roadrunner
{

template<typename State>
CULE_ANNOTATION
void reset(State& s)
{
    s.m_reward   = 0;
    s.m_score    = 0;
    s.tiaFlags.clear(FLAG_ALE_TERMINAL);
    s.m_lives 	 = 3;
}

template<typename State>
CULE_ANNOTATION
void step(State& s)
{
    using cule::atari::games::getDecimalScore;
    using cule::atari::ram::read;

    // update the reward
    int score = 0, mult = 1;
    for (int digit = 0; digit < 4; digit++)
    {
        int value = ram::read(s, 0xC9 + digit) & 0xF;

        // 0xA represents '0, don't display'
        if (value == 0xA) value = 0;
        score += mult * value;
        mult *= 10;
    }
    score *= 100;
    int reward = score - s.m_score;
    s.m_reward = reward;
    s.m_score = score;

    // update terminal status
    int lives_byte = ram::read(s, 0xC4) & 0x7;
    int y_vel = ram::read(s, 0xB9);
    int x_vel_death = ram::read(s, 0xBD);

    s.tiaFlags.template change<FLAG_ALE_TERMINAL>((lives_byte == 0) && ((y_vel != 0) || (x_vel_death != 0)));

    s.m_lives = lives_byte + 1;
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
    int lives_byte = cule::atari::ram::read(s.ram, 0xC4) & 0x7;
    return lives_byte + 1;
}

template<typename State>
CULE_ANNOTATION
 void setTerminal(State& s)
{
    // update terminal status
    int lives_byte = cule::atari::ram::read(s.ram, 0xC4) & 0x7;
    int y_vel = cule::atari::ram::read(s.ram, 0xB9);
    int x_vel_death = cule::atari::ram::read(s.ram, 0xBD);

    s.tiaFlags.template change<FLAG_ALE_TERMINAL>((lives_byte == 0) && ((y_vel != 0) || (x_vel_death != 0)));
}

template<typename State>
CULE_ANNOTATION
int32_t score(State& s)
{
    // update the reward
    int m_score = 0, mult = 1;
    for (int digit = 0; digit < 4; digit++)
    {
        int value = cule::atari::ram::read(s.ram, 0xC9 + digit) & 0xF;

        // 0xA represents '0, don't display'
        if (value == 0xA) value = 0;
        m_score += mult * value;
        mult *= 10;
    }
    m_score *= 100;

    return m_score;
}

template<typename State>
CULE_ANNOTATION
int32_t reward(State& s)
{
    return score(s) - s.score;
}

} // end namespace roadrunner
} // end namespace games
} // end namespace atari
} // end namespace cule


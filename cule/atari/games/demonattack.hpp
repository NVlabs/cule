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
namespace demonattack
{

template<typename State>
CULE_ANNOTATION
void reset(State& s)
{
    s.m_reward   = 0;
    s.m_score    = 0;
    s.tiaFlags.clear(FLAG_ALE_TERMINAL);
    s.m_lives		 = 4;
}

template<typename State>
CULE_ANNOTATION
void step(State& s)
{
    using cule::atari::games::getDecimalScore;
    using cule::atari::ram::read;

    // update the reward
    int score = getDecimalScore(s, 0x85, 0x83, 0x81);

    // MGB: something funny with the RAM; it is not initialized to 0?
    if (ram::read(s, 0x81) == 0xAB &&
            ram::read(s, 0x83) == 0xCD &&
            ram::read(s, 0x85) == 0xEA)
    {
        score = 0;
    }
    s.m_reward = score - s.m_score;
    s.m_score = score;

    // update terminal status
    int lives_displayed = ram::read(s, 0xF2);
    int display_flag = ram::read(s, 0xF1);
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>((lives_displayed == 0) && (display_flag == 0xBD));
    s.m_lives = lives_displayed + 1; // Once we reach terminal, lives() will correctly return 0
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
    case ACTION_RIGHTFIRE:
    case ACTION_LEFTFIRE:
        return true;
    default:
        return false;
    }
}

template<typename State>
CULE_ANNOTATION
int32_t lives(State& s)
{
    return cule::atari::ram::read(s.ram, 0xF2) + 1; // Once we reach terminal, lives() will correctly return 0
}

template<typename State>
CULE_ANNOTATION
void setTerminal(State& s)
{
    // update terminal status
    int lives_displayed = lives(s) - 1;
    int display_flag = cule::atari::ram::read(s.ram, 0xF1);
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>((lives_displayed == 0) && (display_flag == 0xBD));
}

template<typename State>
CULE_ANNOTATION
int32_t score(State& s)
{
    int32_t m_score = cule::atari::games::getDecimalScore(s, 0x85, 0x83, 0x81);

    // MGB: something funny with the RAM; it is not initialized to 0?
    if (cule::atari::ram::read(s.ram, 0x81) == 0xAB &&
        cule::atari::ram::read(s.ram, 0x83) == 0xCD &&
        cule::atari::ram::read(s.ram, 0x85) == 0xEA)
    {
        m_score = 0;
    }

    return m_score;
}

template<typename State>
CULE_ANNOTATION
int32_t reward(State& s)
{
    return score(s) - s.score;
}

} // end namespace demonattack
} // end namespace games
} // end namespace atari
} // end namespace cule


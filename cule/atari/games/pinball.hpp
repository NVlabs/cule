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
namespace pinball
{

template<typename State>
CULE_ANNOTATION
 void reset(State& s)
{
    s.m_reward   = 0;
    s.m_score 	 = 0;
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
    int score = getDecimalScore(s, 0xB0, 0xB2, 0xB4);
    int reward = score - s.m_score;
    s.m_reward = reward;
    s.m_score = score;

    // update terminal status
    int flag = ram::read(s, 0xAF) & 0x1;
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>(flag != 0);

    // The lives in video pinball are displayed as ball number; so #1 == 3 lives
    int lives_byte = ram::read(s, 0x99) & 0x7;
    // And of course, we keep the 'extra ball' counter in a different memory location
    int extra_ball = ram::read(s, 0xA8) & 0x1;

    s.m_lives = 4 + extra_ball - lives_byte;
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
    case ACTION_UPFIRE:
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
    // The lives in video pinball are displayed as ball number; so #1 == 3 lives
    int32_t lives_byte = cule::atari::ram::read(s.ram, 0x99) & 0x7;
    // And of course, we keep the 'extra ball' counter in a different memory location
    int32_t extra_ball = cule::atari::ram::read(s.ram, 0xA8) & 0x1;

    return 4 + extra_ball - lives_byte;
}

template<typename State>
CULE_ANNOTATION
 void setTerminal(State& s)
{
    // update terminal status
    int flag = cule::atari::ram::read(s.ram, 0xAF) & 0x1;
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>(flag != 0);
}

template<typename State>
CULE_ANNOTATION
int32_t score(State& s)
{
    return cule::atari::games::getDecimalScore(s, 0xB0, 0xB2, 0xB4);
}

template<typename State>
CULE_ANNOTATION
int32_t reward(State& s)
{
    return score(s) - s.score;
}

} // end namespace pinball
} // end namespace games
} // end namespace atari
} // end namespace cule


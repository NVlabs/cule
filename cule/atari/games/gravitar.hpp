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
namespace gravitar
{

template<typename State>
CULE_ANNOTATION
void reset(State& s)
{
    s.m_reward   = 0;
    s.m_score    = 0;
    s.tiaFlags.clear(FLAG_ALE_TERMINAL);
    s.m_lives 	 = 6;
}

template<typename State>
CULE_ANNOTATION
void step(State& s)
{
    using cule::atari::games::getDecimalScore;
    using cule::atari::ram::read;

    // update the reward
    int score = getDecimalScore(s, 9, 8, 7);
    int reward = score - s.m_score;
    s.m_reward = reward;
    s.m_score = score;

    // Byte 0x81 contains information about the current screen
    int screen_byte = ram::read(s, 0x81);

    // update terminal status
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>(screen_byte == 0x01);

    // On the starting screen, we set our lives total to 6; otherwise read it from data
    s.m_lives = screen_byte == 0x0? 6 : (ram::read(s, 0x84) + 1);
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
    // Byte 0x81 contains information about the current screen
    int screen_byte = cule::atari::ram::read(s.ram, 0x81);

    // On the starting screen, we set our lives total to 6; otherwise read it from data
    return (screen_byte == 0x0) ? 6 : (cule::atari::ram::read(s.ram, 0x84) + 1);
}

template<typename State>
CULE_ANNOTATION
void setTerminal(State& s)
{
    // Byte 0x81 contains information about the current screen
    int screen_byte = cule::atari::ram::read(s.ram, 0x81);

    // update terminal status
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>(screen_byte == 0x01);
}

template<typename State>
CULE_ANNOTATION
int32_t score(State& s)
{
    return cule::atari::games::getDecimalScore(s, 9, 8, 7);
}

template<typename State>
CULE_ANNOTATION
int32_t reward(State& s)
{
    return score(s) - s.score;
}

} // end namespace gravitar
} // end namespace games
} // end namespace atari
} // end namespace cule


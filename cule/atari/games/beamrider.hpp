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
namespace beamrider
{

template<typename State>
CULE_ANNOTATION
void reset(State& s)
{
    s.m_reward   = 0;
    s.m_score    = 0;
    s.tiaFlags.clear(FLAG_ALE_TERMINAL);
    s.m_lives    = 3;
}

template<typename State>
CULE_ANNOTATION
void step(State& s)
{
    using cule::atari::games::getDecimalScore;
    using cule::atari::ram::read;

    // update the reward
    int score = getDecimalScore(s, 9, 10, 11);
    s.m_reward = score - s.m_score;
    s.m_score = score;
    int new_lives = ram::read(s, 0x85) + 1;

    // Decrease lives *after* the death animation; this is necessary as the lives counter
    // blinks during death
    if (new_lives == s.m_lives - 1)
    {
        if (ram::read(s, 0x8C) == 0x01)
        {
            s.m_lives = new_lives;
        }
    }
    else
    {
        s.m_lives = new_lives;
    }

    // update terminal status
    int byte_val = ram::read(s, 5);
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>(byte_val == 255);
    byte_val = byte_val & 15;
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>(s.tiaFlags[FLAG_ALE_TERMINAL] || (byte_val < 0));
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
    case ACTION_UPRIGHT:
    case ACTION_UPLEFT:
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
    return cule::atari::ram::read(s.ram, 0x85) + 1;
}

template<typename State>
CULE_ANNOTATION
void setTerminal(State& s)
{
    // update terminal status
    int byte_val = cule::atari::ram::read(s.ram, 5);
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>(byte_val == 255);
    byte_val = byte_val & 0xF;
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>(s.tiaFlags[FLAG_ALE_TERMINAL] || (byte_val < 0));
}

template<typename State>
CULE_ANNOTATION
int32_t score(State& s)
{
    return cule::atari::games::getDecimalScore(s, 9, 10, 11);
}

template<typename State>
CULE_ANNOTATION
int32_t reward(State& s)
{
    return score(s) - s.score;
}

} // end namespace beamrider
} // end namespace games
} // end namespace atari
} // end namespace cule


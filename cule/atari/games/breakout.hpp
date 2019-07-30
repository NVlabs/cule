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
namespace breakout
{

// reset
template<typename State>
CULE_ANNOTATION
void reset(State& s)
{
    s.m_reward  = 0;
    s.m_score   = 0;
    s.m_lives   = 5;
    s.tiaFlags.clear(FLAG_ALE_STARTED);
    s.tiaFlags.clear(FLAG_ALE_TERMINAL);
}

// process the latest information from ALE
template<typename State>
CULE_ANNOTATION
void step(State& s)
{
    // update the reward
    uint8_t x = cule::atari::ram::read(s, 77);
    uint8_t y = cule::atari::ram::read(s, 76);

    uint32_t score = 1 * (x & 0x0F) + 10 * ((x & 0xF0) >> 4) + 100 * (y & 0x0F);
    s.m_reward = score - s.m_score;
    s.m_score = score;

    // update terminal status
    s.m_lives = cule::atari::ram::read(s, 57);

    if (!s.tiaFlags[FLAG_ALE_STARTED] && (s.m_lives == 5))
    {
        s.tiaFlags.set(FLAG_ALE_STARTED);
    }

    s.tiaFlags.template change<FLAG_ALE_TERMINAL>(s.tiaFlags[FLAG_ALE_STARTED] && (s.m_lives == 0));
}

// is an action part of the minimal set?
CULE_ANNOTATION
bool isMinimal(const Action& a)
{
    switch (a)
    {
    case ACTION_NOOP:
    case ACTION_FIRE:
    case ACTION_RIGHT:
    case ACTION_LEFT:
        return true;
    default:
        return false;
    }
}

template<typename State>
CULE_ANNOTATION
int32_t lives(State& s)
{
    return cule::atari::ram::read(s.ram, 57);
}

template<typename State>
CULE_ANNOTATION
void setTerminal(State& s)
{
    int m_lives = lives(s);

    if (!s.tiaFlags[FLAG_ALE_STARTED] && (m_lives == 5))
    {
        s.tiaFlags.set(FLAG_ALE_STARTED);
    }

    s.tiaFlags.template change<FLAG_ALE_TERMINAL>(s.tiaFlags[FLAG_ALE_STARTED] && (m_lives == 0));
}

template<typename State>
CULE_ANNOTATION
int32_t score(State& s)
{
    // update the reward
    uint8_t x = cule::atari::ram::read(s.ram, 77);
    uint8_t y = cule::atari::ram::read(s.ram, 76);

    return 1 * (x & 0x0F) + 10 * ((x & 0xF0) >> 4) + 100 * (y & 0x0F);
}

template<typename State>
CULE_ANNOTATION
int32_t reward(State& s)
{
    return score(s) - s.score;
}

} // end namespace breakout
} // end namespace games
} // end namespace atari
} // end namespace cule


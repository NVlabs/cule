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
namespace amidar
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
    int score = getDecimalScore(s, 0xD9, 0xDA, 0xDB);
    s.m_reward = score - s.m_score;
    s.m_score = score;

    // update terminal status
    int livesByte = ram::read(s, 0xD6);

    // MGB it takes one step for the system to reset; this assumes we've
    //  reset
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>(livesByte == 0x80);
    s.m_lives = (livesByte & 0xF);
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
    case ACTION_DOWNFIRE:
        return true;
    default:
        return false;
    }
}

template<typename State>
CULE_ANNOTATION
int32_t lives(State& s)
{
    return cule::atari::ram::read(s.ram, 0xD6) & 0xF;
}

template<typename State>
CULE_ANNOTATION
void setTerminal(State& s)
{
    // MGB it takes one step for the system to reset; this assumes we've reset
    const int32_t livesByte = cule::atari::ram::read(s.ram, 0xD6);
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>(livesByte == 0x80);
}

template<typename State>
CULE_ANNOTATION
int32_t score(State& s)
{
    return cule::atari::games::getDecimalScore(s, 0xD9, 0xDA, 0xDB);
}

template<typename State>
CULE_ANNOTATION
int32_t reward(State& s)
{
    return score(s) - s.score;
}

} // end namespace amidar
} // end namespace games
} // end namespace atari
} // end namespace cule


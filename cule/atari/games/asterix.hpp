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
namespace asterix
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
    int score = getDecimalScore(s, 0xE0, 0xDF, 0xDE);
    s.m_reward = score - s.m_score;
    s.m_score = score;

    // update terminal status
    s.m_lives = ram::read(s, 0xD3) & 0xF;
    int death_counter = ram::read(s, 0xC7);

    // we cannot wait for lives to be set to 0, because the agent has the
    // option of the restarting the game on the very last frame (when lives==1
    // and death_counter == 0x01) by holding 'fire'
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>((death_counter == 0x01) && (s.m_lives == 1));
}

CULE_ANNOTATION
bool isMinimal(const Action &a)
{
    switch (a)
    {
    case ACTION_NOOP:
    case ACTION_UP:
    case ACTION_RIGHT:
    case ACTION_LEFT:
    case ACTION_DOWN:
    case ACTION_UPRIGHT:
    case ACTION_UPLEFT:
    case ACTION_DOWNRIGHT:
    case ACTION_DOWNLEFT:
        return true;
    default:
        return false;
    }
}

template<typename State>
CULE_ANNOTATION
int32_t lives(State& s)
{
    return cule::atari::ram::read(s.ram, 0xD3) & 0xF;
}

template<typename State>
CULE_ANNOTATION
void setTerminal(State& s)
{
    // update terminal status
    int death_counter = cule::atari::ram::read(s.ram, 0xC7);

    // we cannot wait for lives to be set to 0, because the agent has the
    // option of the restarting the game on the very last frame (when lives==1
    // and death_counter == 0x01) by holding 'fire'
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>((death_counter == 0x01) && (lives(s) == 1));
}

template<typename State>
CULE_ANNOTATION
int32_t score(State& s)
{
    return cule::atari::games::getDecimalScore(s, 0xE0, 0xDF, 0xDE);
}

template<typename State>
CULE_ANNOTATION
int32_t reward(State& s)
{
    return score(s) - s.score;
}

} // end namespace asterix
} // end namespace games
} // end namespace atari
} // end namespace cule


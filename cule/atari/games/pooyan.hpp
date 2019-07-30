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
namespace pooyan
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
    int score = getDecimalScore(s, 0x8A, 0x89, 0x88);
    int reward = score - s.m_score;
    s.m_reward = reward;
    s.m_score = score;

    // update terminal status
    int lives_byte = ram::read(s, 0x96);
    int some_byte  = ram::read(s, 0x98);
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>((lives_byte == 0x0) && (some_byte == 0x05));

    s.m_lives = (lives_byte & 0x7) + 1;
}

CULE_ANNOTATION
bool isMinimal(const Action &a)
{
    switch (a)
    {
    case ACTION_NOOP:
    case ACTION_FIRE:
    case ACTION_UP:
    case ACTION_DOWN:
    case ACTION_UPFIRE:
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
    // update terminal status
    int lives_byte = cule::atari::ram::read(s.ram, 0x96);
    return (lives_byte & 0x7) + 1;
}

template<typename State>
CULE_ANNOTATION
 void setTerminal(State& s)
{
    // update terminal status
    int32_t lives_byte = cule::atari::ram::read(s.ram, 0x96);
    int32_t some_byte  = cule::atari::ram::read(s.ram, 0x98);
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>((lives_byte == 0x0) && (some_byte == 0x05));
}

template<typename State>
CULE_ANNOTATION
int32_t score(State& s)
{
    return cule::atari::games::getDecimalScore(s, 0x8A, 0x89, 0x88);
}

template<typename State>
CULE_ANNOTATION
int32_t reward(State& s)
{
    return score(s) - s.score;
}

} // end namespace pooyan
} // end namespace games
} // end namespace atari
} // end namespace cule


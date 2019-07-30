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
namespace tutankham
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
    int score = getDecimalScore(s, 0x9C, 0x9A);
    int reward = score - s.m_score;
    s.m_reward = reward;
    s.m_score = score;

    // update terminal status
    int lives_byte = ram::read(s, 0x9E);
    // byte 0x81 is set to 0x84 when the game is loaded, but not reset
    int some_byte = ram::read(s, 0x81);

    s.tiaFlags.template change<FLAG_ALE_TERMINAL>((lives_byte == 0) && (some_byte != 0x84));

    s.m_lives = (lives_byte & 0x3);
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
    // update terminal status
    int lives_byte = cule::atari::ram::read(s.ram, 0x9E);
    return (lives_byte & 0x3);
}

template<typename State>
CULE_ANNOTATION
 void setTerminal(State& s)
{
    // update terminal status
    int lives_byte = cule::atari::ram::read(s.ram, 0x9E);
    // byte 0x81 is set to 0x84 when the game is loaded, but not reset
    int some_byte = cule::atari::ram::read(s.ram, 0x81);

    s.tiaFlags.template change<FLAG_ALE_TERMINAL>((lives_byte == 0) && (some_byte != 0x84));
}

template<typename State>
CULE_ANNOTATION
int32_t score(State& s)
{
    return cule::atari::games::getDecimalScore(s, 0x9C, 0x9A);
}

template<typename State>
CULE_ANNOTATION
int32_t reward(State& s)
{
    return score(s) - s.score;
}

} // end namespace tutankham
} // end namespace games
} // end namespace atari
} // end namespace cule


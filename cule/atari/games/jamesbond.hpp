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
namespace jamesbond
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
    int score = getDecimalScore(s, 0xDC, 0xDD, 0xDE);
    int reward = score - s.m_score;
    s.m_reward = reward;
    s.m_score = score;

    // update terminal status
    int lives_byte = ram::read(s, 0x86) & 0xF;
    int screen_byte = ram::read(s, 0x8C);

    // byte 0x8C is 0x68 when we die; it does not remain so forever, as
    // the system loops back to start state after a while (where fire will
    // start a new game)
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>((lives_byte == 0) && (screen_byte == 0x68));
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
int32_t lives(State&)
{
    return 0;
}

template<typename State>
CULE_ANNOTATION
void setTerminal(State& s)
{
    using cule::atari::ram::read;

    // update terminal status
    int lives_byte = ram::read(s.ram, 0x86) & 0xF;
    int screen_byte = ram::read(s.ram, 0x8C);

    // byte 0x8C is 0x68 when we die; it does not remain so forever, as
    // the system loops back to start state after a while (where fire will
    // start a new game)
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>((lives_byte == 0) && (screen_byte == 0x68));
}

template<typename State>
CULE_ANNOTATION
int32_t score(State& s)
{
    return cule::atari::games::getDecimalScore(s, 0xDC, 0xDD, 0xDE);
}

template<typename State>
CULE_ANNOTATION
int32_t reward(State& s)
{
    return score(s) - s.score;
}

} // end namespace jamesbond
} // end namespace games
} // end namespace atari
} // end namespace cule


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
namespace alien
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
    int b1 = getDigit(s, 0x8B);
    int b2 = getDigit(s, 0x89);
    int b3 = getDigit(s, 0x87);
    int b4 = getDigit(s, 0x85);
    int b5 = getDigit(s, 0x83);
    int score = b1 + b2 * 10 + b3 * 100 + b4 * 1000 + b5 * 10000;
    score *= 10;
    s.m_reward = score - s.m_score;
    s.m_score = score;

    // update terminal status
    int byte = ram::read(s, 0xC0);
    byte = byte & 15;
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>(byte <= 0);
    s.m_lives = byte;
}

template<typename State>
CULE_ANNOTATION
int getDigit(State& s, int address)
{
    int byte = cule::atari::ram::read(s.ram, address);
    return byte == 0x80 ? 0 : byte >> 3;
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
    return cule::atari::ram::read(s.ram, 0xC0) & 0xF;
}

template<typename State>
CULE_ANNOTATION
void setTerminal(State& s)
{
    // update terminal status
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>(lives(s) <= 0);
}

template<typename State>
CULE_ANNOTATION
int32_t score(State& s)
{
    // update the reward
    int32_t b1 = getDigit(s, 0x8B);
    int32_t b2 = getDigit(s, 0x89);
    int32_t b3 = getDigit(s, 0x87);
    int32_t b4 = getDigit(s, 0x85);
    int32_t b5 = getDigit(s, 0x83);
    int32_t m_score = b1 + b2 * 10 + b3 * 100 + b4 * 1000 + b5 * 10000;
    m_score *= 10;

    return m_score;
}

template<typename State>
CULE_ANNOTATION
int32_t reward(State& s)
{
    return score(s) - s.score;
}

} // end namespace alien
} // end namespace games
} // end namespace atari
} // end namespace cule


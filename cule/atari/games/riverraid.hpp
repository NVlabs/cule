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
namespace riverraid
{

template<typename State>
CULE_ANNOTATION
 void reset(State& s)
{
    s.m_reward   = 0;
    s.m_score    = 0;
    s.tiaFlags.clear(FLAG_ALE_TERMINAL);
    s.m_lives 	 = 0;
    s.m_lives_byte = 0x58;
}

template<typename State>
CULE_ANNOTATION
void step(State& s)
{
    using cule::atari::games::getDecimalScore;
    using cule::atari::ram::read;

    // update the reward
    int score = 0;
    int digit = ram::read(s, 87) / 8;
    score += digit;
    digit = ram::read(s, 85) / 8;
    score += 10 * digit;
    digit = ram::read(s, 83) / 8;
    score += 100 * digit;
    digit = ram::read(s, 81) / 8;
    score += 1000 * digit;
    digit = ram::read(s, 79) / 8;
    score += 10000 * digit;
    digit = ram::read(s, 77) / 8;
    score += 100000 * digit;
    s.m_reward = score - s.m_score;
    s.m_score = score;

    // update terminal status
    int byte_val = ram::read(s, 0xC0);
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>((byte_val == 0x58) && (s.m_lives_byte == 0x59));
    s.m_lives_byte = byte_val;
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
    return cule::atari::ram::read(s.ram, 0xC0);
}

template<typename State>
CULE_ANNOTATION
 void setTerminal(State& s)
{
    // update terminal status
    int byte_val = cule::atari::ram::read(s.ram, 0xC0);
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>((byte_val == 0x58) && s.tiaFlags[FLAG_ALE_STARTED]);
    s.tiaFlags.template change<FLAG_ALE_STARTED>(byte_val == 0x59);
}

template<typename State>
CULE_ANNOTATION
int32_t score(State& s)
{
    // update the reward
    int32_t m_score = 0;
    int32_t m_digit = cule::atari::ram::read(s.ram, 87) / 8;

    m_score += m_digit;
    m_digit = cule::atari::ram::read(s.ram, 85) / 8;
    m_score += 10 * m_digit;
    m_digit = cule::atari::ram::read(s.ram, 83) / 8;
    m_score += 100 * m_digit;
    m_digit = cule::atari::ram::read(s.ram, 81) / 8;
    m_score += 1000 * m_digit;
    m_digit = cule::atari::ram::read(s.ram, 79) / 8;
    m_score += 10000 * m_digit;
    m_digit = cule::atari::ram::read(s.ram, 77) / 8;
    m_score += 100000 * m_digit;

    return m_score;
}

template<typename State>
CULE_ANNOTATION
int32_t reward(State& s)
{
    return score(s) - s.score;
}

} // end namespace riverraid
} // end namespace games
} // end namespace atari
} // end namespace cule


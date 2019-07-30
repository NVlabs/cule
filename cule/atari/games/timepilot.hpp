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
namespace timepilot
{

template<typename State>
CULE_ANNOTATION
void reset(State& s)
{
    s.m_reward   = 0;
    s.m_score 	 = 0;
    s.tiaFlags.clear(FLAG_ALE_TERMINAL);
    s.m_lives 	 = 5;
}

template<typename State>
CULE_ANNOTATION
void step(State& s)
{
    using cule::atari::games::getDecimalScore;
    using cule::atari::ram::read;

    // update the reward
    int score = getDecimalScore(s, 0x8D, 0x8F);
    score *= 100;
    int reward = score - s.m_score;
    s.m_reward = reward;
    s.m_score = score;

    int lives_byte = ram::read(s, 0x8B) & 0x7;
    int screen_byte = ram::read(s, 0x80) & 0xF;

    // update terminal status
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>(ram::read(s, 0xA0));
    // Only update lives when actually flying; otherwise funny stuff happens
    s.m_lives = (screen_byte == 2) ? (lives_byte + 1) : s.m_lives;
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
    int lives_byte = cule::atari::ram::read(s.ram, 0x8B) & 0x7;
    return lives_byte + 1;
}

template<typename State>
CULE_ANNOTATION
 void setTerminal(State& s)
{
    // update terminal status
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>(cule::atari::ram::read(s.ram, 0xA0));
}

template<typename State>
CULE_ANNOTATION
int32_t score(State& s)
{
    return 100 * cule::atari::games::getDecimalScore(s, 0x8D, 0x8F);
}

template<typename State>
CULE_ANNOTATION
int32_t reward(State& s)
{
    return score(s) - s.score;
}

} // end namespace timepilot
} // end namespace games
} // end namespace atari
} // end namespace cule


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
namespace wizard
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
    int score = getDecimalScore(s, 0x86, 0x88);
    if (score >= 8000) score -= 8000; // MGB score does not go beyond 999
    score *= 100;
    s.m_reward = score - s.m_score;
    s.m_score = score;

    // update terminal status
    int newLives = ram::read(s, 0x8D) & 15;
    int byte1 = ram::read(s, 0xF4);

    bool isWaiting = (ram::read(s, 0xD7) & 0x1) == 0;

    s.tiaFlags.template change<FLAG_ALE_TERMINAL>((newLives == 0) && (byte1 == 0xF8));

    // Wizard of Wor decreases the life total when we move into the play field; we only
    // change the life total when we actually are waiting
    s.m_lives = isWaiting ? newLives : s.m_lives;
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
    return cule::atari::ram::read(s.ram, 0x8D) & 15;
}

template<typename State>
CULE_ANNOTATION
void setTerminal(State& s)
{
    // update terminal status
    int newLives = cule::atari::ram::read(s.ram, 0x8D) & 15;
    int byte1 = cule::atari::ram::read(s.ram, 0xF4);

    s.tiaFlags.template change<FLAG_ALE_TERMINAL>((newLives == 0) && (byte1 == 0xF8));
}

template<typename State>
CULE_ANNOTATION
int32_t score(State& s)
{
    // update the reward
    int m_score = cule::atari::games::getDecimalScore(s, 0x86, 0x88);
    if (m_score >= 8000) m_score -= 8000; // MGB score does not go beyond 999
    m_score *= 100;

    return m_score;
}

template<typename State>
CULE_ANNOTATION
int32_t reward(State& s)
{
    return score(s) - s.score;
}

} // end namespace wizard
} // end namespace games
} // end namespace atari
} // end namespace cule


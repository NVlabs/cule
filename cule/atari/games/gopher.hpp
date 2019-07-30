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
namespace gopher
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
    int score = getDecimalScore(s, 0xB2, 0xB1, 0xB0);
    int reward = score - s.m_score;
    s.m_reward = reward;
    s.m_score = score;

    // update terminal status
    int carrot_bits = ram::read(s, 0xB4) & 0x7;
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>(carrot_bits == 0);

    // A very crude popcount
    static int livesFromCarrots[] = { 0, 1, 1, 2, 1, 2, 2, 3};
    s.m_lives = livesFromCarrots[carrot_bits];
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
    // A very crude popcount
    uint32_t carrot_bits = cule::atari::ram::read(s.ram, 0xB4) & 0x7;
    static int livesFromCarrots[] = { 0, 1, 1, 2, 1, 2, 2, 3};
    return livesFromCarrots[carrot_bits];
}

template<typename State>
CULE_ANNOTATION
void setTerminal(State& s)
{
    // update terminal status
    int carrot_bits = cule::atari::ram::read(s.ram, 0xB4) & 0x7;
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>(carrot_bits == 0);
}

template<typename State>
CULE_ANNOTATION
int32_t score(State& s)
{
    return cule::atari::games::getDecimalScore(s, 0xB2, 0xB1, 0xB0);
}

template<typename State>
CULE_ANNOTATION
int32_t reward(State& s)
{
    return score(s) - s.score;
}

} // end namespace gopher
} // end namespace games
} // end namespace atari
} // end namespace cule


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
namespace fishingderby
{

template<typename State>
CULE_ANNOTATION
void reset(State& s)
{
    s.m_reward   = 0;
    s.m_score    = 0;
    s.tiaFlags.clear(FLAG_ALE_TERMINAL);
}
template<typename State>
CULE_ANNOTATION
void step(State& s)
{
    using cule::atari::games::getDecimalScore;
    using cule::atari::ram::read;

    // update the reward
    int my_score   = max(getDecimalScore(s, 0xBD), 0);
    int oppt_score = max(getDecimalScore(s, 0xBE), 0);
    int score = my_score - oppt_score;
    s.m_reward = score - s.m_score;
    s.m_score = score;

    // update terminal status
    int my_score_byte      = ram::read(s, 0xBD);
    int my_oppt_score_byte = ram::read(s, 0xBE);

    s.tiaFlags.template change<FLAG_ALE_TERMINAL>((my_score_byte == 0x99) || (my_oppt_score_byte == 0x99));
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
    // update terminal status
    int my_score_byte      = cule::atari::ram::read(s.ram, 0xBD);
    int my_oppt_score_byte = cule::atari::ram::read(s.ram, 0xBE);

    s.tiaFlags.template change<FLAG_ALE_TERMINAL>((my_score_byte == 0x99) || (my_oppt_score_byte == 0x99));
}

template<typename State>
CULE_ANNOTATION
int32_t score(State& s)
{
    int my_score   = max(cule::atari::games::getDecimalScore(s, 0xBD), 0);
    int oppt_score = max(cule::atari::games::getDecimalScore(s, 0xBE), 0);
    return my_score - oppt_score;
}

template<typename State>
CULE_ANNOTATION
int32_t reward(State& s)
{
   return score(s) - s.score;
}

} // end namespace fishingderby
} // end namespace games
} // end namespace atari
} // end namespace cule


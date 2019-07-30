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
namespace atlantis
{

template<typename State>
CULE_ANNOTATION
void reset(State& s)
{
    s.m_reward   = 0;
    s.m_score    = 0;
    s.tiaFlags.clear(FLAG_ALE_TERMINAL);
    s.m_lives    = 6;
}

template<typename State>
CULE_ANNOTATION
void step(State& s)
{
    using cule::atari::games::getDecimalScore;
    using cule::atari::ram::read;

    // update the reward. Score in Atlantis is a bit funky: when you "roll" the score, it increments
    // the *lowest* digit. E.g., 999900 -> 000001.
    int score = getDecimalScore(s, 0xA2, 0xA3, 0xA1);
    score *= 100;
    s.m_reward = score - s.m_score;

    int old_score = s.m_score;
    s.m_score = score;

    // update terminal status
    s.m_lives = ram::read(s, 0xF1);
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>(s.m_lives == 0xFF);

    //when the game terminates, some garbage gets written on a1, screwing up the score computation
    //since it is not possible to score on the very last frame, we can safely set the reward to 0.
    if(s.tiaFlags[FLAG_ALE_TERMINAL])
    {
        s.m_reward = 0;
        s.m_score = old_score;
    }
}

CULE_ANNOTATION
bool isMinimal(const Action &a)
{
    switch (a)
    {
    case ACTION_NOOP:
    case ACTION_FIRE:
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
    return cule::atari::ram::read(s.ram, 0xF1);
}

template<typename State>
CULE_ANNOTATION
void setTerminal(State& s)
{
    // update terminal status
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>(lives(s) == 0xFF);
}

template<typename State>
CULE_ANNOTATION
int32_t score(State& s)
{
    // update the reward. Score in Atlantis is a bit funky: when you "roll" the score, it increments
    // the *lowest* digit. E.g., 999900 -> 000001.
    return 100 * cule::atari::games::getDecimalScore(s, 0xA2, 0xA3, 0xA1);
}

template<typename State>
CULE_ANNOTATION
int32_t reward(State& s)
{
    int32_t m_reward = score(s) - s.score;

    /* int old_score = s.score; */
    /* s.score = m_score; */

    //when the game terminates, some garbage gets written on a1, screwing up the score computation
    //since it is not possible to score on the very last frame, we can safely set the reward to 0.
    if(s.tiaFlags[FLAG_ALE_TERMINAL])
    {
        m_reward = 0;
        /* s.score = old_score; */
    }

    return m_reward;
}

} // end namespace atlantis
} // end namespace games
} // end namespace atari
} // end namespace cule


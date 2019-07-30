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
namespace qbert
{

template<typename State>
CULE_ANNOTATION
 void reset(State& s)
{
    s.m_reward     = 0;
    s.m_score      = 0;
    s.tiaFlags.clear(FLAG_ALE_TERMINAL);
    s.m_lives 	   = 4;
    s.m_last_lives = 2;
}

template<typename State>
CULE_ANNOTATION
 void step(State& s)
{
    using cule::atari::games::getDecimalScore;
    using cule::atari::ram::read;

    // update terminal status
    int lives_value = ram::read(s, 0x88);
    // Lives start at 2 (4 lives, 3 displayed) and go down to 0xFE (death)
    // Alternatively we can die and reset within one frame; we catch this case
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>((lives_value == 0xFE) || ((lives_value == 0x02) && (s.m_last_lives == -1)));

    // Convert char into a signed integer
    int livesAsChar = static_cast<char>(lives_value);

    if ((s.m_last_lives - 1) == livesAsChar) s.m_lives--;
    s.m_last_lives = livesAsChar;

    // update the reward
    // Ignore reward if reset the game via the fire button; otherwise the agent
    //  gets a big negative reward on its last step
    if (!s.tiaFlags[FLAG_ALE_TERMINAL])
    {
        int score = getDecimalScore(s, 0xDB, 0xDA, 0xD9);
        int reward = score - s.m_score;
        s.m_reward = reward;
        s.m_score = score;
    }
    else
    {
        s.m_reward = 0;
    }
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
        return true;
    default:
        return false;
    }
}

template<typename State>
CULE_ANNOTATION
int32_t lives(State& s)
{
    return cule::atari::ram::read(s.ram, 0x88);
}

template<typename State>
CULE_ANNOTATION
 void setTerminal(State& s)
{
    // update terminal status
    int lives_value = cule::atari::ram::read(s.ram, 0x88);
    // Lives start at 2 (4 lives, 3 displayed) and go down to 0xFE (death)
    // Alternatively we can die and reset within one frame; we catch this case
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>((lives_value == 0xFE) || ((lives_value == 0x02) && s.tiaFlags[FLAG_ALE_STARTED]));

    int livesAsChar = static_cast<char>(lives_value);
    s.tiaFlags.template change<FLAG_ALE_STARTED>(livesAsChar == -1);
}

template<typename State>
CULE_ANNOTATION
int32_t score(State& s)
{
    return cule::atari::games::getDecimalScore(s, 0xDB, 0xDA, 0xD9);
}

template<typename State>
CULE_ANNOTATION
int32_t reward(State& s)
{
    int32_t m_reward = 0;

    // update the reward
    // Ignore reward if reset the game via the fire button; otherwise the agent
    //  gets a big negative reward on its last step
    if (!s.tiaFlags[FLAG_ALE_TERMINAL])
    {
        m_reward = score(s) - s.score;
    }

    return m_reward;
}

} // end namespace qbert
} // end namespace games
} // end namespace atari
} // end namespace cule


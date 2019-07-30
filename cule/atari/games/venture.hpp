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
namespace venture
{

template<typename State>
CULE_ANNOTATION
 void reset(State& s)
{
    s.m_reward   = 0;
    s.m_score 	 = 0;
    s.tiaFlags.clear(FLAG_ALE_TERMINAL);
    s.m_lives 	 = 4;
}

template<typename State>
CULE_ANNOTATION
void step(State& s)
{
    using cule::atari::games::getDecimalScore;
    using cule::atari::ram::read;

    // update the reward
    int score = getDecimalScore(s, 0xC8, 0xC7);
    score *= 100;
    int reward = score - s.m_score;
    s.m_reward = reward;
    s.m_score = score;

    // update terminal status
    int lives_byte = ram::read(s, 0xC6);
    int audio_byte = ram::read(s, 0xCD);
    int death_byte = ram::read(s, 0xBF);
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>((lives_byte == 0) && (audio_byte == 0xFF) && (death_byte & 0x80));

    s.m_lives = (lives_byte & 0x7) + 1;
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
    // update terminal status
    int lives_byte = cule::atari::ram::read(s.ram, 0xC6);
    return (lives_byte & 0x7) + 1;
}

template<typename State>
CULE_ANNOTATION
 void setTerminal(State& s)
{
    // update terminal status
    int lives_byte = cule::atari::ram::read(s.ram, 0xC6);
    int audio_byte = cule::atari::ram::read(s.ram, 0xCD);
    int death_byte = cule::atari::ram::read(s.ram, 0xBF);
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>((lives_byte == 0) && (audio_byte == 0xFF) && (death_byte & 0x80));
}

template<typename State>
CULE_ANNOTATION
int32_t score(State& s)
{
    return 100 * cule::atari::games::getDecimalScore(s, 0xC8, 0xC7);
}

template<typename State>
CULE_ANNOTATION
int32_t reward(State& s)
{
    return score(s) - s.score;
}

} // end namespace venture
} // end namespace games
} // end namespace atari
} // end namespace cule


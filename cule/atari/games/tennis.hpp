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
namespace tennis
{

enum : uint32_t
{
    FIELD_TENNIS_POINTS = 0x000000FF,
    FIELD_TENNIS_SCORE  = 0x00FF0000,
};

template<typename State>
CULE_ANNOTATION
void reset(State& s)
{
    s.m_reward   = 0;
    s.tiaFlags.clear(FLAG_ALE_TERMINAL);
    s.m_prev_delta_points = 0;
    s.m_prev_delta_score = 0;
}

template<typename State>
CULE_ANNOTATION
void step(State& s)
{
    using cule::atari::games::getDecimalScore;
    using cule::atari::ram::read;

    // update the reward
    int my_score     = ram::read(s, 0xC5);
    int oppt_score   = ram::read(s, 0xC6);
    int my_points    = ram::read(s, 0xC7);
    int oppt_points  = ram::read(s, 0xC8);
    int delta_score  = my_score - oppt_score;
    int delta_points = my_points - oppt_points;

    // a reward for the game
    if (s.m_prev_delta_points != delta_points)
        s.m_reward = delta_points - s.m_prev_delta_points;
    // a reward for each point
    else if (s.m_prev_delta_score != delta_score)
        s.m_reward = delta_score - s.m_prev_delta_score;
    else
        s.m_reward = 0;

    s.m_prev_delta_points = delta_points;
    s.m_prev_delta_score = delta_score;

    // update terminal status
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>((my_points >= 6 && delta_points >= 2)    ||
            (oppt_points >= 6 && -delta_points >= 2) ||
            (my_points == 7 || oppt_points == 7));
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
    // update the reward
    const int my_points    = cule::atari::ram::read(s.ram, 0xC7);
    const int oppt_points  = cule::atari::ram::read(s.ram, 0xC8);
    const int delta_points = my_points - oppt_points;

    // update terminal status
    const bool done = (my_points >= 6 && delta_points >= 2) ||
                      (oppt_points >= 6 && -delta_points >= 2) ||
                      (my_points == 7 || oppt_points == 7);
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>(done);
}

template<typename State>
CULE_ANNOTATION
int32_t score(State& s)
{
    const int my_points = cule::atari::ram::read(s.ram, 0xC7);
    const int oppt_points = cule::atari::ram::read(s.ram, 0xC8);
    const int delta_points = my_points - oppt_points;

    const int my_score = cule::atari::ram::read(s.ram, 0xC5);
    const int oppt_score = cule::atari::ram::read(s.ram, 0xC6);
    const int delta_score = my_score - oppt_score;

    int32_t m_score = 0;

    int16_t *ptr = (int16_t*) &m_score;
    ptr[0] = delta_points;
    ptr[1] = delta_score;

    return m_score;
}

template<typename State>
CULE_ANNOTATION
int32_t reward(State& s)
{
    const int my_points = cule::atari::ram::read(s.ram, 0xC7);
    const int oppt_points = cule::atari::ram::read(s.ram, 0xC8);
    const int delta_points = my_points - oppt_points;

    const int my_score = cule::atari::ram::read(s.ram, 0xC5);
    const int oppt_score = cule::atari::ram::read(s.ram, 0xC6);
    const int delta_score = my_score - oppt_score;

    int reward = 0;

    int16_t *ptr = (int16_t*) &s.score;
    const int32_t prev_delta_points = ptr[0];
    const int32_t prev_delta_score  = ptr[1];
    // const int prev_delta_points = SELECT_FIELD(s.score, FIELD_TENNIS_POINTS);
    // const int prev_delta_score  = SELECT_FIELD(s.score, FIELD_TENNIS_SCORE);

    if(prev_delta_points != delta_points)
        reward = delta_points - prev_delta_points;
    else if(prev_delta_score != delta_score)
        reward = delta_score - prev_delta_score;

    return reward;
}

} // end namespace tennis
} // end namespace games
} // end namespace atari
} // end namespace cule


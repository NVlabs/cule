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
namespace adventure
{

template<typename State>
CULE_ANNOTATION
void reset(State& s)
{
    s.m_reward = 0;
    s.tiaFlags.clear(FLAG_ALE_TERMINAL);
}

template<typename State>
CULE_ANNOTATION
void step(State& s)
{
    using cule::atari::ram::read;

    int chalice_status = ram::read(s, 0xB9);
    bool chalice_in_yellow_castle = chalice_status == 0x12;

    if (chalice_in_yellow_castle)
    {
        s.m_reward = 1;
    }

    int player_status = ram::read(s, 0xE0);
    bool player_eaten = player_status == 2;

    if (player_eaten)
    {
        s.m_reward = -1;
    }

    s.tiaFlags.template change<FLAG_ALE_TERMINAL>(player_eaten || chalice_in_yellow_castle);
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
    int chalice_status = cule::atari::ram::read(s.ram, 0xB9);
    bool chalice_in_yellow_castle = chalice_status == 0x12;

    int player_status = cule::atari::ram::read(s.ram, 0xE0);
    bool player_eaten = player_status == 2;

    s.tiaFlags.template change<FLAG_ALE_TERMINAL>(player_eaten || chalice_in_yellow_castle);
}

template<typename State>
CULE_ANNOTATION
int32_t score(State&)
{
    return 0;
}

template<typename State>
CULE_ANNOTATION
int32_t reward(State& s)
{
    int32_t m_reward = 0;

    int chalice_status = cule::atari::ram::read(s.ram, 0xB9);
    bool chalice_in_yellow_castle = chalice_status == 0x12;

    if (chalice_in_yellow_castle)
    {
        m_reward = 1;
    }

    int player_status = cule::atari::ram::read(s.ram, 0xE0);
    bool player_eaten = player_status == 2;
    if (player_eaten)
    {
        m_reward = -1;
    }

    return m_reward;
}

} // end namespace adventure
} // end namespace games
} // end namespace atari
} // end namespace cule


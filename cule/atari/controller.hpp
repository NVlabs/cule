#pragma once

#include <cule/config.hpp>

#include <cule/atari/actions.hpp>
#include <cule/atari/joystick.hpp>
#include <cule/atari/paddles.hpp>

#include <cule/atari/games/detail/types.hpp>

namespace cule
{
namespace atari
{

struct controller
{

template<typename State>
static
CULE_ANNOTATION
void set_flags(State& s,
               const bool& use_paddles,
               const bool& swap_paddles,
               const bool& left_difficulty_B,
               const bool& right_difficulty_B)
{
    s.sysFlags.template change<FLAG_CON_PADDLES>(use_paddles);
    s.sysFlags.template change<FLAG_CON_SWAP>(swap_paddles);

    s.sysFlags.template change<FLAG_SW_LEFT_DIFFLAG_A>(!left_difficulty_B);
    s.sysFlags.template change<FLAG_SW_RIGHT_DIFFLAG_A>(!right_difficulty_B);
}

template<typename State>
static
CULE_ANNOTATION
void set_action(State& s, const Action& player_a_action)
{
    UPDATE_FIELD(s.sysFlags.asBitField(), FIELD_SYS_CON_RESET, player_a_action);
}

template<typename State>
static
CULE_ANNOTATION
void set_actions(State& s, const Action& player_a_action, const Action&)
{
    set_action(s, player_a_action);
}

template<typename State>
static
CULE_ANNOTATION
Action get_action(State& s)
{
    return Action(SELECT_FIELD(s.sysFlags.asBitField(), FIELD_SYS_CON_RESET));
}

template<typename State>
static
CULE_ANNOTATION
void reset(State& s)
{
    UPDATE_FIELD(s.sysFlags.asBitField(), FIELD_SYS_CON, 0);

    if(s.sysFlags[FLAG_CON_PADDLES])
        paddles::reset(s);
    else
        joystick::reset(s);
}

template<typename State>
static
CULE_ANNOTATION
void applyAction(State& s)
{
    // Handle reset
    s.sysFlags.template change<FLAG_SW_RESET_OFF>(!s.sysFlags[FLAG_CON_RESET]);

    if(s.sysFlags[FLAG_CON_PADDLES])
        paddles::applyAction(s);
    else
        joystick::applyAction(s);
}

template<typename State>
static
CULE_ANNOTATION
bool read(State& s, const Control_Jack& jack, const Control_DigitalPin& pin)
{
    bool value = false;

    if(s.sysFlags[FLAG_CON_PADDLES])
        value = paddles::read(s, jack, pin);
    else
        value = joystick::read(s, jack, pin);

    return value;
}

template<typename State>
static
CULE_ANNOTATION
int32_t read(State& s, const Control_Jack& jack, const Control_AnalogPin& pin)
{
    int32_t value = 0;

    if(s.sysFlags[FLAG_CON_PADDLES])
        value = paddles::read(s, jack, pin);
    else
        value = joystick::read(s, jack, pin);

    return value;
}

}; // end namespace controller

} // end namespace atari
} // end namespace cule


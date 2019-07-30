#pragma once

#include <cule/config.hpp>
#include <cule/macros.hpp>

#include <cule/atari/flags.hpp>

#define PADDLE_DELTA 23000

// MGB Values taken from Paddles.cxx (Stella 3.3) - 1400000 * [5,235] / 255
#define PADDLE_MIN 27450

// MGB - was 1290196; updated to 790196... seems to be fine for breakout and pong;
//  avoids pong paddle going off screen
#define PADDLE_MAX 790196

#define PADDLE_DEFAULT_VALUE (((PADDLE_MAX - PADDLE_MIN) / 2) + PADDLE_MIN)

namespace cule
{
namespace atari
{
namespace paddles
{

template<typename State>
CULE_ANNOTATION
 void reset(State& s)
{
    s.resistance = PADDLE_DEFAULT_VALUE;
}

/** Applies paddle actions. This actually modifies the game State by updating the paddle
  *  resistances. */
template<typename State>
CULE_ANNOTATION
 void applyAction(State& s)
{
    // First compute whether we should increase or decrease the paddle position
    s.resistance += (s.sysFlags[FLAG_CON_LEFT] - s.sysFlags[FLAG_CON_RIGHT]) * PADDLE_DELTA;

    // Now update the paddle position
    s.resistance = cule::max(cule::min(s.resistance, PADDLE_MAX), PADDLE_MIN);
}

/**
  Read the value of the specified digital pin for this controller.

  @param pin The pin of the controller jack to read
  @return The State of the pin
*/
template<typename State>
CULE_ANNOTATION
 bool read(State& s, const Control_Jack& jack, const Control_DigitalPin& pin)
{
    return (jack == Control_Right) ||
           ((pin != Control_Three) && (pin != Control_Four)) ||
           (s.sysFlags[FLAG_CON_SWAP] == (pin == Control_Four)) ||
           (s.sysFlags[FLAG_CON_FIRE]==0);
}

/**
  Read the resistance at the specified analog pin for this controller.
  The returned value is the resistance measured in ohms.

  @param pin The pin of the controller jack to read
  @return The resistance at the specified pin
*/
template<typename State>
CULE_ANNOTATION
 int32_t read(State& s, const Control_Jack& jack, const Control_AnalogPin& pin)
{
    return (jack != Control_Right) * ((s.sysFlags[FLAG_CON_SWAP] == (pin == Control_Five)) ? s.resistance : PADDLE_DEFAULT_VALUE);
}

} // end namespace paddles
} // end namespace atari
} // end namespace cule


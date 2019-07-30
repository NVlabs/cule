#pragma once

#include <cule/config.hpp>

#include <cule/macros.hpp>

#include <cule/atari/flags.hpp>

namespace cule
{
namespace atari
{
namespace joystick
{

template<typename State>
CULE_ANNOTATION
void reset(State&){}

template<typename State>
CULE_ANNOTATION
void applyAction(State&){}

/**
  Read the value of the specified digital pin for this controller.

  @param pin The pin of the controller jack to read
  @return The State of the pin
*/
template<typename State>
CULE_ANNOTATION
bool read(State& s, const Control_Jack& jack, const Control_DigitalPin& pin)
{
    if(s.sysFlags[FLAG_CON_SWAP] == (jack == Control_Left)) return true;

    switch(pin)
    {
        case Control_One:
            return s.sysFlags[FLAG_CON_UP] == 0;

        case Control_Two:
            return s.sysFlags[FLAG_CON_DOWN] == 0;

        case Control_Three:
            return s.sysFlags[FLAG_CON_LEFT] == 0;

        case Control_Four:
            return s.sysFlags[FLAG_CON_RIGHT] == 0;

        case Control_Six:
            return s.sysFlags[FLAG_CON_FIRE] == 0;
    }

    return false;
}

/**
  Read the resistance at the specified analog pin for this controller.
  The returned value is the resistance measured in ohms.

  @param pin The pin of the controller jack to read
  @return The resistance at the specified pin
*/
template<typename State>
CULE_ANNOTATION
int32_t read(State&, const Control_Jack&, const Control_AnalogPin&)
{
    return Control_maximumResistance;
}

} // end namespace joystick
} // end namespace atari
} // end namespace cule


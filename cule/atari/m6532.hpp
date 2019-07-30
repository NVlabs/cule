#pragma once

#include <cule/config.hpp>
#include <cule/macros.hpp>

#include <cule/atari/debug.hpp>
#include <cule/atari/flags.hpp>
#include <cule/atari/internals.hpp>
#include <cule/atari/ram.hpp>

#include <cule/atari/types/types.hpp>

#include <iomanip>
#include <cstdlib>

namespace cule
{
namespace atari
{

CULE_ANNOTATION
uint32_t hash(const uint32_t i, const uint32_t seed = 69784899)
{
    unsigned int h = (unsigned int) i ^ (unsigned int) seed;
    h = ~h + (h << 15);
    h =  h ^ (h >> 12);
    h =  h + (h <<  2);
    h =  h ^ (h >>  4);
    h =  h + (h <<  3) + (h << 11);
    h =  h ^ (h >> 16);
    return uint32_t(h);
}

template<typename Controller_t>
struct m6532
{

template<typename State_t>
static
CULE_ANNOTATION
void resetSwitch(State_t& s)
{
    UPDATE_FIELD(s.sysFlags.asBitField(), FIELD_SYS_SW_NODIFF, 0xFF);
}

template<typename State_t>
static
CULE_ANNOTATION
void reset(State_t& s)
{
    // Zero the I/O registers
    UPDATE_FIELD(s.riotData, FIELD_RIOT_DDRA, 0);
    UPDATE_FIELD(s.riotData, FIELD_RIOT_SHIFT, 6);

    UPDATE_FIELD(s.riotData, FIELD_RIOT_TIMER, uint8_t(s.rand & 0xFF));
    s.rand = hash(s.rand);

    s.tiaFlags.clear(FLAG_RIOT_READ_INT);

    s.cyclesWhenTimerSet = 0;
    s.cyclesWhenInterruptReset = 0;

    resetSwitch(s);
}

template<typename State_t>
static
CULE_ANNOTATION
void systemCyclesReset(State_t& s)
{
    // System cycles are being reset to zero so we need to adjust
    // the cycle count we remembered when the timer was last set
    s.cyclesWhenTimerSet -= s.cpuCycles;
    s.cyclesWhenInterruptReset -= s.cpuCycles;
}

template<typename State_t>
static
CULE_ANNOTATION
uint8_t read(State_t& s, const maddr_t& addr)
{
    uint8_t value=0;

    if((addr & 0x0200) == 0x00)
    {
        value = ram::read(s.ram, addr);
    }
    else
    {
        switch(addr & 0x07)
        {
            case 0x00:    // Port A I/O Register (Joystick)
            {
                if(Controller_t::read(s,Control_Left,Control_One))
                    value |= 0x10;
                if(Controller_t::read(s,Control_Left,Control_Two))
                    value |= 0x20;
                if(Controller_t::read(s,Control_Left,Control_Three))
                    value |= 0x40;
                if(Controller_t::read(s,Control_Left,Control_Four))
                    value |= 0x80;

                if(Controller_t::read(s,Control_Right,Control_One))
                    value |= 0x01;
                if(Controller_t::read(s,Control_Right,Control_Two))
                    value |= 0x02;
                if(Controller_t::read(s,Control_Right,Control_Three))
                    value |= 0x04;
                if(Controller_t::read(s,Control_Right,Control_Four))
                    value |= 0x08;
                break;
            }
            case 0x01:    // Port A Data Direction Register
            {
                value = SELECT_FIELD(s.riotData, FIELD_RIOT_DDRA);
                break;
            }
            case 0x02:    // Port B I/O Register (Console switches)
            {
                value = SELECT_FIELD(s.sysFlags.asBitField(), FIELD_SYS_SW);
                break;
            }
            case 0x03:    // Port B Data Direction Register
            {
                break;
            }
            case 0x04:    // Timer Output
            case 0x06:
            {
                uint8_t localTimer = SELECT_FIELD(s.riotData, FIELD_RIOT_TIMER);
                uint8_t intervalShift = SELECT_FIELD(s.riotData, FIELD_RIOT_SHIFT);

                uint32_t cycles = s.cpuCycles - 1;
                uint32_t delta = cycles - s.cyclesWhenTimerSet;
                int32_t timer = (int32_t)localTimer - (int32_t)(delta >> intervalShift) - 1;

                // See if the timer has expired yet?
                if(timer >= 0)
                {
                    value = (uint8_t)timer;
                }
                else
                {
                    timer = (int32_t)(localTimer << intervalShift) - (int32_t)delta - 1;

                    if((timer <= -2) && !s.tiaFlags[FLAG_RIOT_READ_INT])
                    {
                        // Indicate that timer has been read after interrupt occured
                        s.tiaFlags.set(FLAG_RIOT_READ_INT);
                        s.cyclesWhenInterruptReset = s.cpuCycles;
                    }

                    if(s.tiaFlags[FLAG_RIOT_READ_INT])
                    {
                        int32_t offset = s.cyclesWhenInterruptReset - (s.cyclesWhenTimerSet + (localTimer << intervalShift));
                        timer = (int32_t)localTimer - (int32_t)(delta >> intervalShift) - offset;
                    }

                    value = (uint8_t)timer;
                }
                break;
            }
            case 0x05:    // Interrupt Flag
            case 0x07:
            {
                uint8_t localTimer = SELECT_FIELD(s.riotData, FIELD_RIOT_TIMER);
                uint8_t intervalShift = SELECT_FIELD(s.riotData, FIELD_RIOT_SHIFT);

                uint32_t cycles = s.cpuCycles - 1;
                uint32_t delta = cycles - s.cyclesWhenTimerSet;
                int32_t timer = (int32_t)localTimer - (int32_t)(delta >> intervalShift) - 1;

                if((timer >= 0) || s.tiaFlags[FLAG_RIOT_READ_INT])
                    value = 0x00;
                else
                    value = 0x80;

                break;
            }
            default:
            {
                // ERROR(INVALID_MEMORY_ACCESS, MEMORY_CANT_BE_READ, "addr", valueOf(addr));
                break;
            }
        }
    }

    return value;
}

template<typename State_t>
static
CULE_ANNOTATION
void write(State_t& s, const maddr_t& addr, const uint8_t& value)
{
    if((addr & 0x0200) == 0x00)
    {
        ram::write(s.ram, addr, value);
    }
    else
    {
        switch(addr)
        {
            case ADR_SWACNT:
            {
                UPDATE_FIELD(s.riotData, FIELD_RIOT_DDRA, value);
                break;
            }
            case ADR_TIM1T:  // Write timer divide by 1, shift = 0
            case ADR_TIM8T:  // Write timer divide by 8, shift = 3
            case ADR_TIM64T: // Write timer divide by 64, shift = 6
            case ADR_T1024T: // Write timer divide by 1024, shift = 10
            {
                uint8_t temp = (3 * (addr - ADR_TIM1T)) + (addr == ADR_T1024T);
                UPDATE_FIELD(s.riotData, FIELD_RIOT_SHIFT, temp);
                UPDATE_FIELD(s.riotData, FIELD_RIOT_TIMER, value);
                s.cyclesWhenTimerSet = s.cpuCycles;
                s.tiaFlags.clear(FLAG_RIOT_READ_INT);
                break;
            }
            default:
            {
                // ERROR(INVALID_MEMORY_ACCESS, MEMORY_CANT_BE_WRITTEN, "addr", valueOf(addr));
                break;
            }
        }
    }
}

}; // end namespace m6532

} // end namespace atari
} // end namespace cule


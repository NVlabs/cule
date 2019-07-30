#pragma once

#include <cule/config.hpp>
#include <cule/macros.hpp>

#include <cule/atari/internals.hpp>
#include <cule/atari/types/types.hpp>

namespace cule
{
namespace atari
{

template<typename MMC_t,
         typename Stack_t>
struct interrupt
{

enum : uint16_t
{
    VECTOR_NMI=uint16_t(0xFFFA),
    VECTOR_RESET=uint16_t(0xFFFC),
    VECTOR_IRQ=uint16_t(0xFFFE),
};

template<typename State_t>
static
CULE_ANNOTATION
void clearAll(State_t& s)
{
    UPDATE_FIELD(s.sysFlags.asBitField(), FIELD_SYS_INT, 0x00);
}

template<typename State_t>
static
CULE_ANNOTATION
void clear(State_t& s, const SYS_FLAGS& type)
{
    s.sysFlags.clear(type);
}

template<typename State_t>
static
CULE_ANNOTATION
void request(State_t& s, const SYS_FLAGS& type)
{
    // ERROR_IF(sysFlags[type], ILLEGAL_OPERATION, IRQ_ALREADY_PENDING);

    s.sysFlags.set(type);
}

// get address of interrupt handler
template<typename State_t>
static
CULE_ANNOTATION
maddr_t handler(State_t& s, const SYS_FLAGS& type)
{
    maddr_t vector(0);
    switch (type)
    {
        case FLAG_INT_RST:
            vector=VECTOR_RESET;
            break;
        case FLAG_INT_NMI:
            vector=VECTOR_NMI;
            break;
        case FLAG_INT_IRQ:
        case FLAG_INT_BRK:
            vector=VECTOR_IRQ;
            break;
        default:
            break;
    }
    return MMC_t::fetchWordOperand(s,vector);
}

template<typename State_t>
static
CULE_ANNOTATION
bool pending(State_t& s)
{
    return SELECT_FIELD(s.sysFlags.asBitField(), FIELD_SYS_INT) != 0x00;
}

template<typename State_t>
static
CULE_ANNOTATION
bool pending(State_t& s, const SYS_FLAGS& type)
{
    return s.sysFlags[type];
}

// return the highest-priority IRQ that is currently pending
template<typename State_t>
static
CULE_ANNOTATION
SYS_FLAGS current(State_t& s)
{
    if (s.sysFlags[FLAG_INT_RST]) return FLAG_INT_RST;
    if (s.sysFlags[FLAG_INT_NMI]) return FLAG_INT_NMI;
    if (s.sysFlags[FLAG_INT_IRQ]) return FLAG_INT_IRQ;
    if (s.sysFlags[FLAG_INT_BRK]) return FLAG_INT_BRK;
    return SYS_FLAGS(0);
}

template<typename State_t>
static
CULE_ANNOTATION
void poll(State_t& s)
{
    if (pending(s))
    {
        SYS_FLAGS irq = current(s);
        if ((irq != FLAG_INT_IRQ) || !s.sysFlags[FLAG_INTERRUPT_OFF])
        {
            // process IRQ
            if (irq != FLAG_INT_RST)
            {
                Stack_t::pushPC(s);

                // set or clear Break flag depending on irq type
                if (irq == FLAG_INT_BRK)
                    s.sysFlags.set(FLAG_BREAK);
                else
                    s.sysFlags.clear(FLAG_BREAK);

                // push status
                reg_bit_field_t P(SELECT_FIELD(s.sysFlags.asBitField(), FIELD_SYS_PS));
                Stack_t::pushReg(s,P);
                // disable other interrupts
                s.sysFlags.set(FLAG_INTERRUPT_OFF);
            }
            // jump to interrupt handler
            s.PC = handler(s,irq);
            clear(s,irq);
        }
    }
}

}; // end namespace interrupt

} // end namespace atari
} // end namespace cule


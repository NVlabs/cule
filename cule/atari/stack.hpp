#pragma once

#include <cule/config.hpp>

#include <cule/atari/debug.hpp>
#include <cule/atari/internals.hpp>
#include <cule/atari/types/types.hpp>

namespace cule
{
namespace atari
{

template<typename MMC_t>
struct stack
{

    template<typename State_t>
    static
    CULE_ANNOTATION
    void pushByte(State_t& s, const uint8_t byte)
    {
        #ifdef MONITOR_STACK
        printf("[S] Push 0x%02X to $%02X\n",byte,valueOf(s.SP));
        #endif
        MMC_t::write(s, maddr_t(0x0100 + s.SP), byte);
        dec(s.SP);
    }

    template<typename State_t>
    static
    CULE_ANNOTATION
    void pushReg(State_t& s, const reg_bit_field_t& reg)
    {
        pushByte(s, reg);
    }

    template<typename State_t>
    static
    CULE_ANNOTATION
    void pushWord(State_t& s, const word_t& word)
    {
        #ifdef MONITOR_STACK
        printf("[S] Push 0x%04X to $%02X\n",word,valueOf(s.SP));
        #endif
        // FATAL_ERROR_IF(s.SP.reachMax(), INVALID_MEMORY_ACCESS, ILLEGAL_ADDRESS_WARP);

        uint8_t* word_ptr = (uint8_t*)&word;
        pushByte(s, word_ptr[1]);
        pushByte(s, word_ptr[0]);
    }

    template<typename State_t>
    static
    CULE_ANNOTATION
    void pushPC(State_t& s)
    {
        pushWord(s, s.PC);
    }

    template<typename State_t>
    static
    CULE_ANNOTATION
    uint8_t popByte(State_t& s)
    {
        return MMC_t::read(s, maddr_t(0x0100 + inc(s.SP)));
    }

    template<typename State_t>
    static
    CULE_ANNOTATION
    word_t popWord(State_t& s)
    {
        // FATAL_ERROR_UNLESS(s.SP.belowMax(), INVALID_MEMORY_ACCESS, ILLEGAL_ADDRESS_WARP);

        word_t word;
        uint8_t* word_ptr = (uint8_t*)&word;
        word_ptr[0] = popByte(s);
        word_ptr[1] = popByte(s);

        return word;
    }

    template<typename State_t>
    static
    CULE_ANNOTATION
    void reset(State_t& s)
    {
        // move stack pointer to the top of the stack
        s.SP.selfSetMax();
    }

}; // end namespace stack

} // end namespace atari
} // end namespace cule


#pragma once

#include <cule/config.hpp>
#include <cule/macros.hpp>

#include <cule/atari/m6532.hpp>
#include <cule/atari/internals.hpp>
#include <cule/atari/rom.hpp>
#include <cule/atari/tia.hpp>

#include <cule/atari/types/types.hpp>

namespace cule
{
namespace atari
{

template<typename AccessorType,
         typename M6532Type,
         typename TIAType>
struct mmc
{
    using Accessor_t = AccessorType;
    using M6532_t = M6532Type;
    using TIA_t = TIAType;

    template<typename State_t>
    static
    CULE_ANNOTATION
    uint8_t read(State_t& s, const maddr_t& addr)
    {
        uint8_t value=uint8_t(~0);

        if(addr & 0x1000)
        {
            value = Accessor_t::read(s, addr);
        }
        else if(addr & 0x0080)
        {
            value = M6532_t::read(s, addr);
        }
        else
        {
            value = TIA_t::read(s, addr);
        }

        s.sysFlags.set(FLAG_CPU_LAST_READ);
        s.noise = value;

        return value;
    }

    template<typename State_t>
    static
    CULE_ANNOTATION
    void write(State_t& s, const maddr_t& addr, const uint8_t& value)
    {
        if(addr & 0x1000)
        {
            /* ERROR(INVALID_MEMORY_ACCESS, MEMORY_CANT_BE_WRITTEN, "addr", valueOf(addr), "value", value); */
            Accessor_t::write(s, addr, value);
        }
        else if(addr & 0x0080)
        {
            M6532_t::write(s, addr, value);
        }
        else
        {
            TIA_t::write(s, addr, value);
        }

        s.sysFlags.clear(FLAG_CPU_LAST_READ);
        s.noise = value;
    }

    template<typename State_t>
    static
    CULE_ANNOTATION
    opcode_t fetchOpcode(State_t& s, maddr_t& pc)
    {
        // WARN_IF(!MSB(pc), INVALID_MEMORY_ACCESS, MEMORY_NOT_EXECUTABLE, "PC", valueOf(pc));
        const opcode_t opcode(read(s,pc));
        inc(pc);

        return opcode;
    }

    template<typename State_t>
    static
    CULE_ANNOTATION
    operandb_t fetchByteOperand(State_t& s, maddr_t& pc)
    {
        operandb_t operand(read(s,pc));
        inc(pc);

        return operand;
    }

    template<typename State_t>
    static
    CULE_ANNOTATION
    operandw_t fetchWordOperand(State_t& s, maddr_t& pc)
    {
        // FATAL_ERROR_IF(pc.reachMax(), INVALID_MEMORY_ACCESS, ILLEGAL_ADDRESS_WARP);
        uint8_t a = read(s,pc);
        uint8_t b = read(s,maddr_t(pc + 1));
        operandw_t operand(makeWord(a, b));
        pc += 2;

        return operand;
    }

    template<typename State_t>
    static
    CULE_ANNOTATION
    uint8_t loadZPByte(State_t& s, const maddr_t& zp)
    {
        return read(s,zp);
    }

    template<typename State_t>
    static
    CULE_ANNOTATION
    word_t loadZPWord(State_t& s, const maddr_t& zp)
    {
        // FATAL_ERROR_IF(zp.reachMax(), INVALID_MEMORY_ACCESS, ILLEGAL_ADDRESS_WARP);
        uint8_t a = read(s,zp);
        uint8_t b = read(s,maddr_t(zp + 1));
        return makeWord(a, b);
    }

}; // end namespace mmc

} // end namespace atari
} // end namespace cule


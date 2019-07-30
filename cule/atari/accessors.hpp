#pragma once

#include <cule/config.hpp>

#include <cule/atari/ram.hpp>

#include <cule/atari/types/types.hpp>

namespace cule
{
namespace atari
{

template<int32_t dummy>
struct rom_accessor
{
    template<class State_t>
    static
    CULE_ANNOTATION
    void initialize(State_t&)
    {}

    template<class State_t>
    static
    CULE_ANNOTATION
    void write(State_t&, const maddr_t&, const uint8_t&)
    {}

    template<class State_t>
    static
    CULE_ANNOTATION
    uint8_t read(State_t&, const maddr_t&)
    {
        return 0;
    }
};

template<>
struct rom_accessor<ROM_2K>
{
    template<class State_t>
    static
    CULE_ANNOTATION
    void initialize(State_t&)
    {}

    template<class State_t>
    static
    CULE_ANNOTATION
    void write(State_t&, const maddr_t&, const uint8_t&)
    {}

    template<class State_t>
    static
    CULE_ANNOTATION
    uint8_t read(State_t& s, const maddr_t& addr)
    {
        return s.rom[addr & 0x07FF];
    }
};

template<>
struct rom_accessor<ROM_4K>
{
    template<class State_t>
    static
    CULE_ANNOTATION
    void initialize(State_t&)
    {}

    template<class State_t>
    static
    CULE_ANNOTATION
    void write(State_t&, const maddr_t&, const uint8_t&)
    {}

    template<class State_t>
    static
    CULE_ANNOTATION
    uint8_t read(State_t& s, const maddr_t& addr)
    {
        return s.rom[addr & 0x0FFF];
    }
};

template<>
struct rom_accessor<ROM_F8SC>
{
    template<class State_t>
    static
    CULE_ANNOTATION
    void initialize(State_t& s)
    {
        s.bank = 1;
    }

    template<class State_t>
    static
    CULE_ANNOTATION
    void write(State_t& s, const maddr_t& addr, const uint8_t& value)
    {
        //if(addr < 0x1080)
        if((addr & 0x1FFF) < 0x1100)
        {
            ram::write(s.ram + (128 / 4), addr, value);
        }
        else
        {
            switch(addr & 0x0FFF)
            {
                case 0x0FF8:
                {
                    s.bank = 0;
                    break;
                }
                case 0x0FF9:
                {
                    s.bank = 1;
                    break;
                }
                default:
                {
                    break;
                }
            }
        }
    }

    template<class State_t>
    static
    CULE_ANNOTATION
    uint8_t read(State_t& s, const maddr_t& addr)
    {
        uint8_t value = 0;

        // if((addr >= 0x1080) && (addr < 0x1100))
        if((addr & 0x1FFF) < 0x1100)
        {
            value = ram::read(s.ram + (128 / 4), addr);
        }
        else
        {
            write(s, addr, 0);
            value = s.rom[(s.bank * 4096) + (addr & 0x0FFF)];
        }

        return value;
    }
};

template<>
struct rom_accessor<ROM_F8>
{
    template<class State_t>
    static
    CULE_ANNOTATION
    void initialize(State_t& s)
    {
        s.bank = 1;
    }

    template<class State_t>
    static
    CULE_ANNOTATION
    void write(State_t& s, const maddr_t& addr, const uint8_t&)
    {
        //if(addr < 0x1080)
        switch(addr & 0x0FFF)
        {
            case 0x0FF8:
            {
                s.bank = 0;
                break;
            }
            case 0x0FF9:
            {
                s.bank = 1;
                break;
            }
            default:
            {
                break;
            }
        }
    }

    template<class State_t>
    static
    CULE_ANNOTATION
    uint8_t read(State_t& s, const maddr_t& addr)
    {
        write(s, addr, 0);
        return s.rom[(s.bank * 4096) + (addr & 0x0FFF)];
    }
};

template<>
struct rom_accessor<ROM_FE>
{
    template<class State_t>
    static
    CULE_ANNOTATION
    void initialize(State_t&)
    {}

    template<class State_t>
    static
    CULE_ANNOTATION
    void write(State_t&, const maddr_t&, const uint8_t&)
    {}

    template<class State_t>
    static
    CULE_ANNOTATION
    uint8_t read(State_t& s, const maddr_t& addr)
    {
        return s.rom[(addr & 0x0FFF) + (((addr & 0x2000) == 0) ? 4096 : 0)];
    }
};

template<>
struct rom_accessor<ROM_F6>
{
    template<class State_t>
    static
    CULE_ANNOTATION
    void initialize(State_t& s)
    {
        s.bank = 0;
    }

    template<class State_t>
    static
    CULE_ANNOTATION
    void write(State_t& s, const maddr_t& addr, const uint8_t&)
    {
        switch(addr & 0x0FFF)
        {
            case 0x0FF6:
            {
                s.bank = 0;
                break;
            }
            case 0x0FF7:
            {
                s.bank = 1;
                break;
            }
            case 0x0FF8:
            {
                s.bank = 2;
                break;
            }
            case 0x0FF9:
            {
                s.bank = 3;
                break;
            }
            default:
            {
                break;
            }
        }
    }

    template<class State_t>
    static
    CULE_ANNOTATION
    uint8_t read(State_t& s, const maddr_t& addr)
    {
        write(s, addr, 0);
        return s.rom[(s.bank * 4096) + (addr & 0x0FFF)];
    }
};

template<>
struct rom_accessor<ROM_E0>
{
    template<class State_t>
    static
    CULE_ANNOTATION
    void initialize(State_t& s)
    {
        s.bank = 0x7654;
    }

    template<class State_t>
    static
    CULE_ANNOTATION
    void write(State_t& s, const maddr_t& addr, const uint8_t&)
    {
        maddr_t address(addr & 0x0FFF);

        // Switch banks if necessary
        if((address >= 0x0FE0) && (address <= 0x0FE7))
        {
            s.bank = (s.bank & 0xFFF0) | (address & 0x7);
        }
        else if((address >= 0x0FE8) && (address <= 0x0FEF))
        {
            s.bank = (s.bank & 0xFF0F) | ((address & 0x7) << 4);
        }
        else if((address >= 0x0FF0) && (address <= 0x0FF7))
        {
            s.bank = (s.bank & 0xF0FF) | ((address & 0x7) << 8);
        }
    }

    template<class State_t>
    static
    CULE_ANNOTATION
    uint8_t read(State_t& s, const maddr_t& addr)
    {
        write(s, addr, 0);
        return s.rom[(((s.bank >> (4 * ((addr >> 10) & 0x3))) & 0x7) << 10) + (addr & 0x03FF)];
    }
};

} // end namespace atari
} // end namespace cule


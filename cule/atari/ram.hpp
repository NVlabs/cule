#pragma once

#include <cule/config.hpp>

#include <cule/atari/internals.hpp>

namespace cule
{
namespace atari
{
namespace ram
{

#define RAM_INDEX_ACCESSOR

#ifndef RAM_INDEX_ACCESSOR
template<int INDEX, typename Addr>
CULE_ANNOTATION
void get(uint32_t* ptr, const Addr& addr, uint8_t& value)
{
    value = SELECT_FIELD(ptr[INDEX], (0xFF << (8 * int(addr & 0x03))));
}

template<int INDEX, typename Addr>
CULE_ANNOTATION
void set(uint32_t* ptr, const Addr& addr, const uint8_t& value)
{
    UPDATE_FIELD(ptr[INDEX], (0xFF << (8 * int(addr & 0x03))), value);
}

template<typename Addr>
CULE_ANNOTATION
uint8_t read(uint32_t* ptr, const Addr& addr)
{
    uint8_t value=0;

    switch((addr & 0x7C)>>2)
    {
        case 0:
            get<0>(ptr, addr, value);
            break;
        case 1:
            get<1>(ptr, addr, value);
            break;
        case 2:
            get<2>(ptr, addr, value);
            break;
        case 3:
            get<3>(ptr, addr, value);
            break;
        case 4:
            get<4>(ptr, addr, value);
            break;
        case 5:
            get<5>(ptr, addr, value);
            break;
        case 6:
            get<6>(ptr, addr, value);
            break;
        case 7:
            get<7>(ptr, addr, value);
            break;
        case 8:
            get<8>(ptr, addr, value);
            break;
        case 9:
            get<9>(ptr, addr, value);
            break;
        case 10:
            get<10>(ptr, addr, value);
            break;
        case 11:
            get<11>(ptr, addr, value);
            break;
        case 12:
            get<12>(ptr, addr, value);
            break;
        case 13:
            get<13>(ptr, addr, value);
            break;
        case 14:
            get<14>(ptr, addr, value);
            break;
        case 15:
            get<15>(ptr, addr, value);
            break;
        case 16:
            get<16>(ptr, addr, value);
            break;
        case 17:
            get<17>(ptr, addr, value);
            break;
        case 18:
            get<18>(ptr, addr, value);
            break;
        case 19:
            get<19>(ptr, addr, value);
            break;
        case 20:
            get<20>(ptr, addr, value);
            break;
        case 21:
            get<21>(ptr, addr, value);
            break;
        case 22:
            get<22>(ptr, addr, value);
            break;
        case 23:
            get<23>(ptr, addr, value);
            break;
        case 24:
            get<24>(ptr, addr, value);
            break;
        case 25:
            get<25>(ptr, addr, value);
            break;
        case 26:
            get<26>(ptr, addr, value);
            break;
        case 27:
            get<27>(ptr, addr, value);
            break;
        case 28:
            get<28>(ptr, addr, value);
            break;
        case 29:
            get<29>(ptr, addr, value);
            break;
        case 30:
            get<30>(ptr, addr, value);
            break;
        case 31:
            get<31>(ptr, addr, value);
            break;
    }

    return value;
}

template<typename Addr>
CULE_ANNOTATION
void write(uint32_t* ptr, const Addr& addr, const uint8_t& value)
{
    switch((addr & 0x7C)>>2)
    {
        case 0:
            set<0>(ptr, addr, value);
            break;
        case 1:
            set<1>(ptr, addr, value);
            break;
        case 2:
            set<2>(ptr, addr, value);
            break;
        case 3:
            set<3>(ptr, addr, value);
            break;
        case 4:
            set<4>(ptr, addr, value);
            break;
        case 5:
            set<5>(ptr, addr, value);
            break;
        case 6:
            set<6>(ptr, addr, value);
            break;
        case 7:
            set<7>(ptr, addr, value);
            break;
        case 8:
            set<8>(ptr, addr, value);
            break;
        case 9:
            set<9>(ptr, addr, value);
            break;
        case 10:
            set<10>(ptr, addr, value);
            break;
        case 11:
            set<11>(ptr, addr, value);
            break;
        case 12:
            set<12>(ptr, addr, value);
            break;
        case 13:
            set<13>(ptr, addr, value);
            break;
        case 14:
            set<14>(ptr, addr, value);
            break;
        case 15:
            set<15>(ptr, addr, value);
            break;
        case 16:
            set<16>(ptr, addr, value);
            break;
        case 17:
            set<17>(ptr, addr, value);
            break;
        case 18:
            set<18>(ptr, addr, value);
            break;
        case 19:
            set<19>(ptr, addr, value);
            break;
        case 20:
            set<20>(ptr, addr, value);
            break;
        case 21:
            set<21>(ptr, addr, value);
            break;
        case 22:
            set<22>(ptr, addr, value);
            break;
        case 23:
            set<23>(ptr, addr, value);
            break;
        case 24:
            set<24>(ptr, addr, value);
            break;
        case 25:
            set<25>(ptr, addr, value);
            break;
        case 26:
            set<26>(ptr, addr, value);
            break;
        case 27:
            set<27>(ptr, addr, value);
            break;
        case 28:
            set<28>(ptr, addr, value);
            break;
        case 29:
            set<29>(ptr, addr, value);
            break;
        case 30:
            set<30>(ptr, addr, value);
            break;
        case 31:
            set<31>(ptr, addr, value);
            break;
    }
}
#else
template<typename Addr>
CULE_ANNOTATION
uint8_t read(uint32_t* ptr, const Addr& addr)
{
    return SELECT_FIELD(ptr[(addr & 0x7C)>>2], (0xFF << (8 * int(addr & 0x03))));
}
template<typename Addr>
CULE_ANNOTATION
void write(uint32_t* ptr, const Addr& addr, const uint8_t& value)
{
    UPDATE_FIELD(ptr[(addr & 0x7C)>>2], (0xFF << (8 * int(addr & 0x03))), value);
}
#endif

} // end namespace ram
} // end namespace atari
} // end namespace cule


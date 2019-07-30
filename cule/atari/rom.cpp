#pragma once

#include <cule/config.hpp>
#include <cule/md5.hpp>

#include <cule/atari/ale.hpp>
#include <cule/atari/games.hpp>
#include <cule/atari/ram.hpp>
#include <cule/atari/rom.hpp>
#include <cule/atari/types/types.hpp>

#include <array>
#include <iomanip>
#include <iostream>
#include <string>

#include <cstring>
#include <zlib.h>

namespace cule
{
namespace atari
{
namespace detail
{
bool search_for_bytes(const int32_t size, const uint8_t* data, const uint8_t* signature,
                      const uint32_t sigsize, const uint32_t minhits)
{
    uint32_t count = 0;
    for(uint32_t i = 0; i < size - sigsize; ++i)
    {
        uint32_t matches = 0;
        for(uint32_t j = 0; j < sigsize; ++j)
        {
            if(data[i + j] == signature[j])
                ++matches;
            else
                break;
        }
        if(matches == sigsize)
        {
            ++count;
            i += sigsize;  // skip past this signature 'window' entirely
        }
        if(count >= minhits)
        {
            break;
        }
    }

    return (count >= minhits);
}

bool is_probably_CV(const int32_t size, const uint8_t* data)
{
    // CV RAM access occurs at addresses $f3ff and $f400
    // These signatures are attributed to the MESS project
    uint8_t signature[2][3] =
    {
        { 0x9D, 0xFF, 0xF3 },  // STA $F3FF
        { 0x99, 0x00, 0xF4 }   // STA $F400
    };

    if(search_for_bytes(size, data, signature[0], 3, 1))
        return true;
    else
        return search_for_bytes(size, data, signature[1], 3, 1);
}

bool is_probably_SC(const int32_t size, const uint8_t* data)
{
    // We assume a Superchip cart contains the same bytes for its entire
    // RAM area; obviously this test will fail if it doesn't
    // The RAM area will be the first 256 bytes of each 4K bank
    uint32_t banks = size / 4096;
    for(uint32_t i = 0; i < banks; i++)
    {
        uint8_t first = data[i * 4096];
        for(uint32_t j = 0; j < 256; j++)
        {
            if(data[(i * 4096) + j] != first)
                return false;
        }
    }
    return true;
}

bool is_probably_E0(const int32_t size, const uint8_t* data)
{
    // E0 cart bankswitching is triggered by accessing addresses
    // $FE0 to $FF9 using absolute non-indexed addressing
    // To eliminate false positives (and speed up processing), we
    // search for only certain known signatures
    // Thanks to "stella@casperkitty.com" for this advice
    // These signatures are attributed to the MESS project
    uint8_t signature[6][3] =
    {
        { 0x8D, 0xE0, 0x1F },  // STA $1FE0
        { 0x8D, 0xE0, 0x5F },  // STA $5FE0
        { 0x8D, 0xE9, 0xFF },  // STA $FFE9
        { 0xAD, 0xE9, 0xFF },  // LDA $FFE9
        { 0xAD, 0xED, 0xFF },  // LDA $FFED
        { 0xAD, 0xF3, 0xBF }   // LDA $BFF3
    };
    for(size_t i = 0; i < 6; ++i)
    {
        if(search_for_bytes(size, data, signature[i], 3, 1))
        {
            return true;
        }
    }
    return false;
}

bool is_probably_3E(const int32_t size, const uint8_t* data)
{
    // 3E cart bankswitching is triggered by storing the bank number
    // in address 3E using 'STA $3E', commonly followed by an
    // immediate mode LDA
    uint8_t signature[] = { 0x85, 0x3E, 0xA9, 0x00 };  // STA $3E; LDA #$00
    return search_for_bytes(size, data, signature, 4, 1);
}

bool is_probably_3F(const int32_t size, const uint8_t* data)
{
    // 3F cart bankswitching is triggered by storing the bank number
    // in address 3F using 'STA $3F'
    // We expect it will be present at least 2 times, since there are
    // at least two banks
    uint8_t signature[] = { 0x85, 0x3F };  // STA $3F
    return search_for_bytes(size, data, signature, 2, 2);
}

bool is_probably_UA(const int32_t size, const uint8_t* data)
{
    // UA cart bankswitching switches to bank 1 by accessing address 0x240
    // using 'STA $240'
    uint8_t signature[] = { 0x8D, 0x40, 0x02 };  // STA $240
    return search_for_bytes(size, data, signature, 3, 1);
}

bool is_probably_FE(const int32_t size, const uint8_t* data)
{
    // FE bankswitching is very weird, but always seems to include a
    // 'JSR $xxxx'
    // These signatures are attributed to the MESS project
    uint8_t signature[4][5] =
    {
        { 0x20, 0x00, 0xD0, 0xC6, 0xC5 },  // JSR $D000; DEC $C5
        { 0x20, 0xC3, 0xF8, 0xA5, 0x82 },  // JSR $F8C3; LDA $82
        { 0xD0, 0xFB, 0x20, 0x73, 0xFE },  // BNE $FB; JSR $FE73
        { 0x20, 0x00, 0xF0, 0x84, 0xD6 }   // JSR $F000; STY $D6
    };
    for(uint32_t i = 0; i < 4; ++i)
    {
        if(search_for_bytes(size, data, signature[i], 5, 1))
            return true;
    }
    return false;
}

ROM_FORMAT detect_type(const int32_t size, const uint8_t* data)
{
    ROM_FORMAT rom_type = ROM_NOT_SUPPORTED;

    if((size == 2048) || (size == 4096))
    {
        if(is_probably_CV(size, data))
        {
            rom_type = ROM_CV;
        }
        else
        {
            rom_type = (size == 2048) || (memcmp(data, data + 2048, 2048) == 0) ? ROM_2K : ROM_4K;
        }
    }
    else if(size == 8192)
    {
        if(is_probably_SC(size, data))
            rom_type = ROM_F8SC;
        else if(memcmp(data, data + 4096, 4096) == 0)
            rom_type = ROM_4K;
        else if(is_probably_E0(size, data))
            rom_type = ROM_E0;
        else if(is_probably_3E(size, data))
            rom_type = ROM_3E;
        else if(is_probably_3F(size, data))
            rom_type = ROM_3F;
        else if(is_probably_UA(size, data))
            rom_type = ROM_UA;
        else if(is_probably_FE(size, data))
            rom_type = ROM_FE;
        else
            rom_type = ROM_F8;
    }
    else if(size == 16384)
    {
        rom_type = ROM_F6;
    }

    return rom_type;
}
} // end namespace detail

rom::rom(const std::string& filename)
    : _ram_size(0),
      _rom_size(0)
{
    if(filename.size())
    {
        reset(filename);
    }
}

rom::rom(const rom& other)
    : _ram_size(other._ram_size),
      _rom_size(other._rom_size),
      _type(other._type),
      _gameId(other._gameId),
      _md5(other._md5),
      _filename(other._filename),
      image(other.image),
      _minimal_actions(other._minimal_actions)
{}

void rom::reset(const std::string& filename)
{
    _filename = filename;

    gzFile f = gzopen(_filename.c_str(), "rb");
    CULE_ASSERT(f != nullptr, "gzopen failed to open " << _filename);

    image.resize(MAX_ROM_SIZE);
    _rom_size = gzread(f, image.data(), MAX_ROM_SIZE);
    gzclose(f);

    compute_md5();
    set_game_id();

    _type = detail::detect_type(rom_size(), data());

    _ram_size = (_type == ROM_F8SC) ? 256 : 128;
}

std::vector<Action> const& rom::minimal_actions() const
{
    return _minimal_actions;
}

std::string rom::file_name() const
{
    return _filename;
}

std::string rom::game_name() const
{
    return value_or_default(games::ROM_ATTR_Name);
}

std::string rom::type_name() const
{
    return names[type()];
}

bool rom::swap_ports() const
{
    return value_or_default(games::ROM_ATTR_SwapPorts) == "YES";
}

bool rom::use_paddles() const
{
    return value_or_default(games::ROM_ATTR_ControllerLeft) == "PADDLES";
}

bool rom::swap_paddles() const
{
    return value_or_default(games::ROM_ATTR_SwapPaddles) == "YES";
}

bool rom::allow_hmove_blanks() const
{
    return value_or_default(games::ROM_ATTR_HmoveBlanks) == "YES";
}

bool rom::player_left_difficulty_B() const
{
    return value_or_default(games::ROM_ATTR_LeftDifficulty) == "B";
}

bool rom::player_right_difficulty_B() const
{
    return value_or_default(games::ROM_ATTR_RightDifficulty) == "B";
}

size_t rom::ram_size() const
{
    return _ram_size;
}

size_t rom::rom_size() const
{
    return _rom_size;
}

size_t rom::screen_height() const
{
    return is_ntsc() ? NTSC_SCREEN_HEIGHT : PAL_SCREEN_HEIGHT;
}

size_t rom::screen_width() const
{
    return SCREEN_WIDTH;
}

size_t rom::screen_size() const
{
    return screen_height() * screen_width();
}

ROM_FORMAT rom::type() const
{
    return _type;
}

games::GAME_TYPE rom::game_id() const
{
    return _gameId;
}

std::string rom::md5() const
{
    return _md5;
}

uint8_t const* rom::data() const
{
    return image.data();
}

bool rom::is_supported() const
{
    bool ret = false;

    switch(_type)
    {
    case ROM_2K:
    case ROM_4K:
    case ROM_F8:
    case ROM_F8SC:
    case ROM_E0:
    case ROM_FE:
    case ROM_F6:
    {
        ret = true;
        break;
    }
    default:
    {
        break;
    }
    }

    return ret;
}

bool rom::is_ntsc() const
{
    return game_name().find("PAL") == std::string::npos;
}

std::ostream& operator<< (std::ostream& os, rom const& cart)
{
    os << "Cartridge\n";
    os << "\tFile       : " << cart.file_name() << "\n";
    os << "\tName       : " << cart.game_name() << "\n";
    os << "\tController : " << (cart.use_paddles() ? "Paddles" : "Joystick") << "\n";
    os << "\tSwapped    : " << ((cart.swap_paddles() || cart.swap_ports()) ? "Yes" : "No") << "\n";
    os << "\tLeft  Diff : " << (cart.player_left_difficulty_B()  ? "B" : "A") << "\n";
    os << "\tRight Diff : " << (cart.player_right_difficulty_B() ? "B" : "A") << "\n";
    os << "\tType       : " << cart.type_name() << "\n";
    os << "\tDisplay    : " << (cart.is_ntsc() ? "NTSC" : "PAL") << "\n";
    os << "\tRAM Size   : " << cart.ram_size() << "\n";
    os << "\tROM Size   : " << cart.rom_size() << "\n";
    os << "\tMD5        : " << cart.md5() << "\n";
    return os;
}

std::string rom::value_or_default(const games::ROM_ATTR attr) const
{
    CULE_ASSERT(rom_size() != 0, "Rom has not been properly initialized");

    using namespace games;
    const std::string n = rom_attr_data[game_id()][attr];
    return !n.empty() ? n : default_attr[attr];
}

bool rom::has_banks() const
{
    return (type() == ROM_F8SC) || (type() == ROM_F8) || (type() == ROM_F6) || (type() == ROM_E0);
}

void rom::set_game_id()
{
    _gameId = games::rom_game_map[md5()];
    _minimal_actions = getMinimalActionSet(_gameId);
}

void rom::compute_md5()
{
    // If we get to this point, we know we have a valid file to open
    // Now we make sure that the file has a valid properties entry
    std::array<uint8_t,MD5_DIGEST_LENGTH> md5_result;
    MD5(image.data(), _rom_size, md5_result.data());

    std::ostringstream buffer;
    buffer.fill('0');
    buffer.setf(std::ios::hex, std::ios::basefield);
    for(auto c : md5_result)
    {
        buffer << std::setw(2) << uint32_t(c);
    }
    _md5 = buffer.str();
}

const char* rom::names[_ROM_MAX];
static void initStrings()
{
    rom::names[ROM_2K]="2K";
    rom::names[ROM_4K]="4K";
    rom::names[ROM_CV]="CV";
    rom::names[ROM_F8SC]="F8SC";
    rom::names[ROM_E0]="E0";
    rom::names[ROM_3E]="3E";
    rom::names[ROM_3F]="3F";
    rom::names[ROM_UA]="UA";
    rom::names[ROM_FE]="FE";
    rom::names[ROM_F8]="F8";
    rom::names[ROM_F6]="F6";
    rom::names[ROM_NOT_SUPPORTED]="NOT SUPPORTED";
}

class rom_initialize
{
  public:
    rom_initialize()
    {
        // const strings
        initStrings();
    }
};
rom_initialize startup;

} // end namespace atari
} // end namespace cule


#pragma once

#include <cule/config.hpp>

#include <cule/atari/flags.hpp>
#include <cule/atari/games/detail/attributes.hpp>

#include <string>
#include <vector>

namespace cule
{
namespace atari
{

class rom
{
public:

    enum
    {
        MAX_ROM_SIZE = 16 * 1024,
    };

    static const char* names[_ROM_MAX];

    rom(const std::string& filename = "");

    rom(const rom& other);

    void reset(const std::string& filename);

    std::vector<Action> const& minimal_actions() const;

    std::string file_name() const;

    std::string game_name() const;

    std::string type_name() const;

    std::string md5() const;

    bool swap_ports() const;

    bool use_paddles() const;

    bool swap_paddles() const;

    bool allow_hmove_blanks() const;

    bool player_left_difficulty_B() const;

    bool player_right_difficulty_B() const;

    bool is_supported() const;

    bool is_ntsc() const;

    size_t ram_size() const;

    size_t rom_size() const;

    size_t screen_height() const;

    size_t screen_width() const;

    size_t screen_size() const;

    ROM_FORMAT type() const;

    bool has_banks() const;

    games::GAME_TYPE game_id() const;

    uint8_t const* data() const;

    template<typename State>
    CULE_ANNOTATION
    static void write(State& s, const maddr_t& addr, const uint8_t& value);

    template<typename State>
    CULE_ANNOTATION
    static uint8_t read(State& s, const maddr_t& addr);

private:

    std::string value_or_default(const games::ROM_ATTR attr) const;

    void set_game_id();

    void compute_md5();

    size_t _ram_size;
    size_t _rom_size;
    ROM_FORMAT _type;
    games::GAME_TYPE _gameId;

    std::string _md5;
    std::string _filename;
    std::vector<uint8_t> image;
    std::vector<Action> _minimal_actions;
}; // end rom class

} // end namespace atari
} // end namespace cule


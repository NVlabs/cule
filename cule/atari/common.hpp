#pragma once

#include <cule/config.hpp>

#include <cule/atari/internals.hpp>

namespace cule
{
namespace atari
{

// utils
CULE_ANNOTATION
word_t makeWord(const uint8_t lo, const uint8_t hi)
{
    return word_t(lo) | (word_t(hi) << 8);
}

CULE_ANNOTATION
int16_t clamp(int16_t value)
{
    if(value >= 160)
        value -= 160;
    else if(value < 0)
        value += 160;
    return value;
}

std::string get_frame_name(const size_t proc_id, const size_t frame_index)
{
    std::ostringstream png_filename;
    png_filename << "frames/";
    png_filename << proc_id << "/";
    png_filename << std::setfill('0') << std::setw(6) << frame_index;
    png_filename << ".png";
    return png_filename.str();
}

} // end namespace atari
} // end namespace cule


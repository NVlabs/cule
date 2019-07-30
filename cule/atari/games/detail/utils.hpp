#pragma once

#include <cule/config.hpp>

#include <cule/atari/ram.hpp>
#include <cule/atari/state.hpp>

namespace cule
{
namespace atari
{
namespace games
{

template<typename State>
CULE_ANNOTATION
 int getDecimalScore(State& s, const int idx)
{
    int score = 0;
    int digits_val = ram::read(s.ram, idx);
    int right_digit = digits_val & 15;
    int left_digit = digits_val >> 4;
    score += ((10 * left_digit) + right_digit);

    return score;
}

template<typename State>
CULE_ANNOTATION
 int getDecimalScore(State& s, int lower_index, int higher_index)
{
    int score = 0;
    int lower_digits_val = ram::read(s.ram, lower_index);
    int lower_right_digit = lower_digits_val & 15;
    int lower_left_digit = (lower_digits_val - lower_right_digit) >> 4;
    score += ((10 * lower_left_digit) + lower_right_digit);

    if (higher_index < 0)
    {
        return score;
    }

    int higher_digits_val = ram::read(s.ram, higher_index);
    int higher_right_digit = higher_digits_val & 15;
    int higher_left_digit = (higher_digits_val - higher_right_digit) >> 4;
    score += ((1000 * higher_left_digit) + 100 * higher_right_digit);

    return score;
}

template<typename State>
CULE_ANNOTATION
 int getDecimalScore(State& s, int lower_index, int middle_index, int higher_index)
{
    int score = getDecimalScore(s, lower_index, middle_index);
    int higher_digits_val = ram::read(s.ram, higher_index);
    int higher_right_digit = higher_digits_val & 15;
    int higher_left_digit = (higher_digits_val - higher_right_digit) >> 4;
    score += ((100000 * higher_left_digit) + 10000 * higher_right_digit);

    return score;
}

} // end namespace games
} // end namespace atari
} // end namespace cule


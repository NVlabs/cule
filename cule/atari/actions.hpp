#pragma once

#include <cule/config.hpp>

#include <cule/atari/flags.hpp>

#include <map>
#include <vector>

namespace cule
{
namespace atari
{

static const Action allActions[_ACTION_MAX] =
{
    ACTION_NOOP,
    ACTION_FIRE,
    ACTION_UP,
    ACTION_RIGHT,
    ACTION_LEFT,
    ACTION_DOWN,

    ACTION_UPRIGHT,
    ACTION_UPLEFT,
    ACTION_DOWNRIGHT,
    ACTION_DOWNLEFT,
    ACTION_UPFIRE,
    ACTION_RIGHTFIRE,
    ACTION_LEFTFIRE,
    ACTION_DOWNFIRE,
    ACTION_UPRIGHTFIRE,
    ACTION_UPLEFTFIRE,
    ACTION_DOWNRIGHTFIRE,
    ACTION_DOWNLEFTFIRE,

    ACTION_RESET,
};

static std::map<Action, std::string> action_to_string_map =
{
    {ACTION_NOOP,          "ACTION_NOOP"},
    {ACTION_RIGHT,         "ACTION_RIGHT"},
    {ACTION_LEFT,          "ACTION_LEFT"},
    {ACTION_DOWN,          "ACTION_DOWN"},
    {ACTION_UP,            "ACTION_UP"},
    {ACTION_FIRE,          "ACTION_FIRE"},

    {ACTION_UPRIGHT,       "ACTION_UPRIGHT"},
    {ACTION_UPLEFT,        "ACTION_UPLEFT"},
    {ACTION_DOWNRIGHT,     "ACTION_DOWNRIGHT"},
    {ACTION_DOWNLEFT,      "ACTION_DOWNLEFT"},
    {ACTION_UPFIRE,        "ACTION_UPFIRE"},
    {ACTION_RIGHTFIRE,     "ACTION_RIGHTFIRE"},
    {ACTION_LEFTFIRE,      "ACTION_LEFTFIRE"},
    {ACTION_DOWNFIRE,      "ACTION_DOWNFIRE"},
    {ACTION_UPRIGHTFIRE,   "ACTION_UPRIGHTFIRE"},
    {ACTION_UPLEFTFIRE,    "ACTION_UPLEFTFIRE"},
    {ACTION_DOWNRIGHTFIRE, "ACTION_DOWNRIGHTFIRE"},
    {ACTION_DOWNLEFTFIRE,  "ACTION_DOWNLEFTFIRE"},

    {ACTION_RESET,         "ACTION_RESET"},
};

} // end namespace atari
} // end namespace cule


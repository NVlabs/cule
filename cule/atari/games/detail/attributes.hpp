#pragma once

#include <cule/config.hpp>

namespace cule
{
namespace atari
{
namespace games
{

enum GAME_TYPE : uint8_t
{
    GAME_BOWLING,
    GAME_BOXING,
    GAME_BREAKOUT,
    GAME_FISHING_DERBY,
    GAME_FREEWAY,
    GAME_KABOOM,
    GAME_PONG,
    GAME_SKIING,
    GAME_TENNIS,

    GAME_ADVENTURE,
    GAME_AIR_RAID,
    GAME_ALIEN,
    GAME_AMIDAR,
    GAME_ASSAULT,
    GAME_ATLANTIS,
    GAME_BANK_HEIST,
    GAME_BERZERK,
    GAME_CARNIVAL,
    GAME_CHOPPER,
    GAME_DEFENDER,
    GAME_DEMON_ATTACK,
    GAME_ENDURO,
    GAME_FROSTBITE,
    GAME_GOPHER,
    GAME_ICE_HOCKEY,
    GAME_JOURNEY_ESCAPE,
    GAME_NAME_THIS_GAME,
    GAME_PITFALL,
    GAME_POOYAN,
    GAME_QBERT,
    GAME_RIVERRAID,
    GAME_SEAQUEST,
    GAME_SPACE_INVADERS,
    GAME_STAR_GUNNER,
    GAME_VENTURE,
    GAME_PINBALL,
    GAME_WIZARD,
    GAME_YARS_REVENGE,

    GAME_ASTERIX,
    GAME_ASTEROIDS,
    GAME_BATTLE_ZONE,
    GAME_BEAM_RIDER,
    GAME_CENTIPEDE,
    GAME_CRAZY_CLIMBER,
    GAME_ELEVATOR_ACTION,
    GAME_GRAVITAR,
    GAME_HERO,
    GAME_JAMESBOND,
    GAME_KANGAROO,
    GAME_KRULL,
    GAME_KUNG_FU_MASTER,
    GAME_MONTEZUMA_REVENGE,
    GAME_MS_PACMAN,
    GAME_PHOENIX,
    GAME_PRIVATE_EYE,
    GAME_ROBOTANK,
    GAME_TIME_PILOT,
    GAME_TUTANKHAM,
    GAME_UP_N_DOWN,
    GAME_ZAXXON,

    GAME_DOUBLE_DUNK,
    GAME_ROAD_RUNNER,
    GAME_SOLARIS,
};

enum ROM_ATTR
{
    ROM_ATTR_Manufacturer,
    ROM_ATTR_ModelNo,
    ROM_ATTR_Name,
    ROM_ATTR_Note,
    ROM_ATTR_Rarity,
    ROM_ATTR_Sound,
    ROM_ATTR_Type,
    ROM_ATTR_LeftDifficulty,
    ROM_ATTR_RightDifficulty,
    ROM_ATTR_TelevisionType,
    ROM_ATTR_SwapPorts,
    ROM_ATTR_ControllerLeft,
    ROM_ATTR_ControllerRight,
    ROM_ATTR_SwapPaddles,
    ROM_ATTR_Format,
    ROM_ATTR_YStart,
    ROM_ATTR_Height,
    ROM_ATTR_Phosphor,
    ROM_ATTR_PPBlend,
    ROM_ATTR_HmoveBlanks,
    _ROM_ATTR_MAX
};

} // end namespace games
} // end namespace atari
} // end namespace cule


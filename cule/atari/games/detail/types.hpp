#pragma once

#include <cule/config.hpp>

#include <cule/atari/games/detail/types.hpp>

#include <map>
#include <string>

namespace cule
{
namespace atari
{
namespace games
{

static std::map<std::string, GAME_TYPE> rom_game_map =
{
    // 2K Games
    { "c9b7afad3bfd922e006a6bfc1d4f3fe7", GAME_BOWLING},
    { "c3ef5c4653212088eda54dc91d787870", GAME_BOXING},
    { "f34f08e5eb96e500e851a80be3277a56", GAME_BREAKOUT},
    { "b8865f05676e64f3bec72b9defdacfa7", GAME_FISHING_DERBY},
    { "8e0ab801b1705a740b476b7f588c6d16", GAME_FREEWAY},
    { "dbdaf82f4f0c415a94d1030271a9ef44", GAME_KABOOM},
    { "60e0ea3cbe0913d39803477945e9e5ec", GAME_PONG},
    { "367411b78119299234772c08df10e134", GAME_SKIING},
    { "a5c96b046d5f8b7c96daaa12f925bef8", GAME_TENNIS},

    // 4K Games
    { "4b27f5397c442d25f0c418ccdacf1926", GAME_ADVENTURE},
    { "35be55426c1fec32dfb503b4f0651572", GAME_AIR_RAID},
    { "f1a0a23e6464d954e3a9579c4ccd01c8", GAME_ALIEN},
    { "056f5d886a4e7e6fdd83650554997d0d", GAME_AMIDAR},
    { "c31a17942d162b80962cb1f7571cd1d5", GAME_ASSAULT},
    { "0b33252b680b65001e91a411e56e72e9", GAME_ATLANTIS},
    { "83b8c01c72306d60dd9b753332ebd276", GAME_BANK_HEIST},
    { "fac28963307b6e85082ccd77c88325e7", GAME_BERZERK},
    { "de29e46dbea003c3c09c892d668b9413", GAME_CARNIVAL},
    { "c1cb228470a87beb5f36e90ac745da26", GAME_CHOPPER},
    { "e1029676edb3d35b76ca943da7434da8", GAME_DEFENDER},
    { "b24f6a5820a4b7763a3d547e3e07441d", GAME_DEMON_ATTACK},
    { "94b92a882f6dbaa6993a46e2dcc58402", GAME_ENDURO},
    { "4ca73eb959299471788f0b685c3ba0b5", GAME_FROSTBITE},
    { "8f90590dba143d783df5a6cff2000e4d", GAME_GOPHER},
    { "47711c44723da5d67047990157dcb5dd", GAME_ICE_HOCKEY},
    { "718ae62c70af4e5fd8e932fee216948a", GAME_JOURNEY_ESCAPE},
    { "f98d2276d4a25b286135566255aea9d0", GAME_NAME_THIS_GAME},
    { "f73d2d0eff548e8fc66996f27acf2b4b", GAME_PITFALL},
    { "668dc528b7ea9345140f4fcfbecf7066", GAME_POOYAN},
    { "484b0076816a104875e00467d431c2d2", GAME_QBERT},
    { "da5096000db5fdaa8d02db57d9367998", GAME_RIVERRAID},
    { "240bfbac5163af4df5ae713985386f92", GAME_SEAQUEST},
    { "61dbe94f110f30ca4ec524ae5ce2d026", GAME_SPACE_INVADERS},
    { "a3c1c70024d7aabb41381adbfb6d3b25", GAME_STAR_GUNNER},
    { "c63a98ca404aa5ee9fcff1de488c3f43", GAME_VENTURE},
    { "a2424c1a0c783d7585d701b1c71b5fdc", GAME_PINBALL},
    { "663ef22eb399504d5204c543b8a86bcd", GAME_WIZARD},
    { "ee8027d554d14c8d0b86f94737d2fdcc", GAME_YARS_REVENGE},

    // 8K Games
    { "89a68746eff7f266bbf08de2483abe55", GAME_ASTERIX},
    { "ccbd36746ed4525821a8083b0d6d2c2c", GAME_ASTEROIDS},
    { "e434c0e161dd3c3fb435eb6bad2e182c", GAME_BATTLE_ZONE},
    { "79ab4123a83dc11d468fb2108ea09e2e", GAME_BEAM_RIDER},
    { "17d000a2882f9fdaa8b4a391ad367f00", GAME_CENTIPEDE},
    { "55ef7b65066428367844342ed59f956c", GAME_CRAZY_CLIMBER},
    { "71f8bacfbdca019113f3f0801849057e", GAME_ELEVATOR_ACTION},
    { "4767356fa0ed3ebe21437b4473d4ee28", GAME_GRAVITAR},
    { "fca4a5be1251927027f2c24774a02160", GAME_HERO},
    { "e51030251e440cffaab1ac63438b44ae", GAME_JAMESBOND},
    { "4326edb70ff20d0ee5ba58fa5cb09d60", GAME_KANGAROO},
    { "cc724ebe74a109e39c0b2784ddc980ca", GAME_KRULL},
    { "0b4e793c9425175498f5a65a3e960086", GAME_KUNG_FU_MASTER},
    { "3347a6dd59049b15a38394aa2dafa585", GAME_MONTEZUMA_REVENGE},
    { "9469d18238345d87768e8965f9f4a6b2", GAME_MS_PACMAN},
    { "7e52a95074a66640fcfde124fffd491a", GAME_PHOENIX},
    { "ef3a4f64b6494ba770862768caf04b86", GAME_PRIVATE_EYE},
    { "4f618c2429138e0280969193ed6c107e", GAME_ROBOTANK},
    { "fc2104dd2dadf9a6176c1c1c8f87ced9", GAME_TIME_PILOT},
    { "66c2380c71709efa7b166621e5bb4558", GAME_TUTANKHAM},
    { "a499d720e7ee35c62424de882a3351b6", GAME_UP_N_DOWN},
    { "eea0da9b987d661264cce69a7c13c3bd", GAME_ZAXXON},

    // 16K Games
    { "cfc226d04d7490b69e155abd7741e98c", GAME_DOUBLE_DUNK},
    { "7d3cdde63b16fa637c4484e716839c94", GAME_ROAD_RUNNER},
    { "e72eb8d4410152bdcb69e7fba327b420", GAME_SOLARIS},
};

const std::array<std::string,_ROM_ATTR_MAX> rom_attr_names =
{{
        "Cartridge.Manufacturer",
        "Cartridge.ModelNo",
        "Cartridge.Name",
        "Cartridge.Note",
        "Cartridge.Rarity",
        "Cartridge.Sound",
        "Cartridge.Type",
        "Console.LeftDifficulty",
        "Console.RightDifficulty",
        "Console.TelevisionType",
        "Console.SwapPorts",
        "Controller.Left",
        "Controller.Right",
        "Controller.SwapPaddles",
        "Display.Format",
        "Display.YStart",
        "Display.Height",
        "Display.Phosphor",
        "Display.PPBlend",
        "Emulation.HmoveBlanks"
    }
};

const std::array<std::string,_ROM_ATTR_MAX> default_attr =
{{
        "",            // Cartridge.Manufacturer
        "",            // Cartridge.ModelNo
        "Untitled",    // Cartridge.Name
        "",            // Cartridge.Note
        "",            // Cartridge.Rarity
        "MONO",        // Cartridge.Sound
        "AUTO-DETECT", // Cartridge.Type
        "B",           // Console.LeftDifficulty
        "B",           // Console.RightDifficulty
        "COLOR",       // Console.TelevisionType
        "NO",          // Console.SwapPorts
        "JOYSTICK",    // Controller.Left
        "JOYSTICK",    // Controller.Right
        "NO",          // Controller.SwapPaddles
        "AUTO-DETECT", // Display.Format
        "34",          // Display.YStart
        "210",         // Display.Height
        "NO",          // Display.Phosphor
        "77",          // Display.PPBlend
        "YES"          // Emulation.HmoveBlanks
    }
};

static std::map<GAME_TYPE, std::array<std::string,_ROM_ATTR_MAX>> rom_attr_data =
{
    // 2K Games
    { GAME_BOWLING, {{"Atari", "CX2628 / 6699842 / 4975117", "Bowling (1978) (Atari) [!]", "", "Common", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "" }}},
    { GAME_BOXING, {{"Activision", "AG-002", "Boxing (1981) (Activision) [!]", "", "Uncommon", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "" }}},
    { GAME_BREAKOUT, {{"Atari", "CX2622", "Breakout - Breakaway IV (1978) (Atari)", "Uses the Paddle Controllers", "Common", "", "", "", "", "", "", "PADDLES", "", "", "", "", "", "", "", "" }}},
    { GAME_FISHING_DERBY, {{"Activision", "AG-004", "Fishing Derby (1980) (Activision) [!]", "", "Uncommon", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "" }}},
    { GAME_FREEWAY, {{"Activision", "AG-009", "Freeway (1981) (Activision) [!]", "", "Uncommon", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "" }}},
    { GAME_KABOOM, {{"CCE", "", "Kaboom! (CCE)", "Uses the Paddle Controllers (left only)", "", "", "", "", "", "", "", "PADDLES", "", "", "", "", "", "", "", "" }}},
    { GAME_PONG, {{"Atari", "CX2621", "Video Olympics (1978) (Atari)", "Uses the Paddle Controllers", "Common", "", "", "", "", "", "", "PADDLES", "PADDLES", "YES", "", "", "", "", "", "" }}},
    { GAME_SKIING, {{"Atari", "", "Skiing (32-in-1) (Atari) (PAL) [!]", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "" }}},
    { GAME_TENNIS, {{"Activision", "AG-007", "Tennis (1981) (Activision) (PAL) [!]", "", "Common", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "" }}},

    // 4K Games
    { GAME_ADVENTURE, {{"Atari", "CX2613 / 4975154", "Adventure (1978) (Atari) (PAL) [!]", "", "Common", "", "", "", "", "", "", "", "NONE", "", "", "", "", "", "", "" }}},
    { GAME_AIR_RAID, {{"", "C-817", "Air Raid (Men-A-Vision) (PAL)", "", "Extremely Rare", "", "", "", "", "", "", "", "", "", "", "", "", "YES", "", "" }}},
    { GAME_ALIEN, {{"20th Century Fox", "11006", "Alien (1982) (20th Century Fox)", "", "Rare", "", "", "", "", "", "", "", "", "", "", "", "", "YES", "", "" }}},
    { GAME_AMIDAR, {{"Parker Bros", "PB5310", "Amidar (1983) (Parker Bros) (PAL) [!]", "", "Uncommon", "", "", "A", "A", "", "", "", "NONE", "", "", "", "", "", "", "" }}},
    { GAME_ASSAULT, {{"Rainbow Vision", "", "Monster aus dem All (1983) (Rainbow Vision) (PAL) [!]", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "" }}},
    { GAME_ATLANTIS, {{"CCE", "", "Atlantis (CCE)", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "" }}},
    { GAME_BANK_HEIST, {{"", "", "Bank Heist (PAL)", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "" }}},
    { GAME_BERZERK, {{"CCE", "", "Berzerk (CCE)", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "" }}},
    { GAME_CARNIVAL, {{"CBS Electronics", "", "Carnival (1983) (CBS Electronics) (PAL) [!]", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "" }}},
    { GAME_CHOPPER, {{"Activision", "AX-015", "Chopper Command (1982) (Activision) [!]", "", "Uncommon", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "" }}},
    { GAME_DEFENDER, {{"", "", "Defender", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "NO" }}},
    { GAME_DEMON_ATTACK, {{"", "", "Demon Attack", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "" }}},
    { GAME_ENDURO, {{"Activision", "AX-026", "Enduro (1983) (Activision) [!]", "", "Uncommon", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "" }}},
    { GAME_FROSTBITE, {{"Activision", "AX-031", "Frostbite (1983) (Activision)", "", "Rare", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "" }}},
    { GAME_GOPHER, {{"", "", "Gopher (1982) (PAL)", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "" }}},
    { GAME_ICE_HOCKEY, {{"CCE", "", "Ice Hockey (CCE)", "", "Uncommon", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "" }}},
    { GAME_JOURNEY_ESCAPE, {{"", "112-006", "Journey - Escape (1982) (Data Age) [!]", "", "Common", "", "", "", "", "", "", "", "", "", "", "", "230", "YES", "", "" }}},
    { GAME_NAME_THIS_GAME, {{"", "", "Name This Game", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "" }}},
    { GAME_PITFALL, {{"CCE", "AX-018", "Pitfall! (CCE) (PAL-M) [!]", "", "Common", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "" }}},
    { GAME_POOYAN, {{"Konami-Gakken", "", "Pooyan (1982) (Konami-Gakken) (PAL) [!]", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "" }}},
    { GAME_QBERT, {{"Atari", "CX26150", "Q-bert (1988) (Atari) [!]", "", "Rare", "", "", "", "", "", "", "", "", "", "", "40", "", "", "", "" }}},
    { GAME_RIVERRAID, {{"", "", "River Raid", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "" }}},
    { GAME_SEAQUEST, {{"Activision", "AX-022", "Seaquest (1983) (Activision) [!]", "", "Rare", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "" }}},
    { GAME_SPACE_INVADERS, {{"", "", "Space Invaders", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "" }}},
    { GAME_STAR_GUNNER, {{"Telesys", "1005", "Star Gunner (1982) (Telesys)", "", "Extremely Rare", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "" }}},
    { GAME_VENTURE, {{"Atari", "CX26145", "Venture (1988) (Atari) [!]", "", "Rare", "", "", "", "", "", "", "", "", "", "", "", "", "YES", "", "" }}},
    { GAME_PINBALL, {{"Atari", "", "Video Pinball (1980) (Atari) (PAL) [p1][!]", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "" }}},
    { GAME_WIZARD, {{"CBS Electronics", "M8774", "Wizard of Wor (1982) (CBS Electronics) (PAL) [!]", "Uses the Joystick Controllers (swapped)", "Rare", "", "", "", "", "", "YES", "", "", "", "", "", "", "YES", "", "" }}},
    { GAME_YARS_REVENGE, {{"", "", "Yars Revenge", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "" }}},

    // 8K Games
    { GAME_ASTERIX, {{"Atari", "CX2696", "Asterix (1988) (Atari) (Prototype) (NTSC)", "", "Extremely Rare", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "" }}},
    { GAME_ASTEROIDS, {{"", "CX2649", "Asteroids [p1]", "", "Common", "", "", "", "", "", "", "", "", "", "", "", "", "YES", "", "NO" }}},
    { GAME_BATTLE_ZONE, {{"", "", "Battle Zone", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "" }}},
    { GAME_BEAM_RIDER, {{"Activision", "AZ-037-04", "Beamrider (1983) (Activision) [!]", "", "Rare", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "" }}},
    { GAME_CENTIPEDE, {{"Atari", "CX2676", "Centipede (1982) (Atari) (PAL) [!]", "", "Common", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "" }}},
    { GAME_CRAZY_CLIMBER, {{"", "CX2683", "Crazy Climber (1983) (Atari)", "", "Extremely Rare", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "" }}},
    { GAME_ELEVATOR_ACTION, {{"Atari", "", "Elevator Action (Atari) (Prototype)", "", "Prototype", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "" }}},
    { GAME_GRAVITAR, {{"", "", "Gravitar", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "" }}},
    { GAME_HERO, {{"Activision", "AZ-036-04", "H.E.R.O. (1984) (Activision) [!]", "", "Rare", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "" }}},
    { GAME_JAMESBOND, {{"Parker Bros", "PB5110", "James Bond 007 (1983) (Parker Bros)", "", "Rare", "", "", "", "", "", "", "", "", "", "", "", "", "YES", "", "" }}},
    { GAME_KANGAROO, {{"Atari", "CX2689", "Kangaroo (1983) (Atari)", "", "Uncommon", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "" }}},
    { GAME_KRULL, {{"", "", "Krull", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "" }}},
    { GAME_KUNG_FU_MASTER, {{"CCE", "AX-039", "Kung Fu Master (CCE)", "", "Rare", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "" }}},
    { GAME_MONTEZUMA_REVENGE, {{"Parker Bros", "PB5760", "Montezuma's Revenge - Starring Panama Joe (1983) (Parker Bros)", "", "Extremely Rare", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "" }}},
    { GAME_MS_PACMAN, {{"CCE", "", "Ms. Pac-Man (1982) (CCE)", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "NO" }}},
    { GAME_PHOENIX, {{"Atari", "CX2673", "Phoenix (1982) (Atari)", "", "Common", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "" }}},
    { GAME_PRIVATE_EYE, {{"Activision", "AG-034-04", "Private Eye (1983) (Activision) [!]", "", "Rare", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "" }}},
    { GAME_ROBOTANK, {{"Activision", "AZ-028", "Robot Tank (1983) (Activision) [!]", "", "Rare", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "" }}},
    { GAME_TIME_PILOT, {{"Coleco", "2663", "Time Pilot (1983) (Coleco)", "", "Rare", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "" }}},
    { GAME_TUTANKHAM, {{"Parker Bros", "", "Tutankham (1983) (Parker Bros) (PAL) [!]", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "" }}},
    { GAME_UP_N_DOWN, {{"Sega", "009-01", "Up 'n Down (1983) (Sega)", "", "Extremely Rare", "", "", "", "", "", "", "", "", "", "", "30", "", "", "", "" }}},
    { GAME_ZAXXON, {{"CBS Electronics", "4L-2277", "Zaxxon (1983) (CBS Electronics)", "", "Extremely Rare", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "" }}},

    //16K Games
    { GAME_DOUBLE_DUNK, {{"Atari", "CX26159", "Double Dunk (1989) (Atari) (PAL) [!]", "", "Rare", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "" }}},
    { GAME_ROAD_RUNNER, {{"CCE", "", "Road Runner (CCE)", "", "", "", "", "", "", "", "", "", "", "", "", "20", "", "", "", "" }}},
    { GAME_SOLARIS, {{"Atari", "CX26136", "Solaris (1986) (Atari)", "", "Uncommon", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "" }}}
};

} // end namespace games
} // end namespace atari
} // end namespace cule


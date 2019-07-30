#pragma once

#include <cule/config.hpp>

#include <cstdlib>

namespace cule
{
namespace atari
{

enum ROM_FORMAT : uint8_t
{
    ROM_2K,
    ROM_4K,
    ROM_CV,
    ROM_F8SC,
    ROM_E0,
    ROM_3E,
    ROM_3F,
    ROM_UA,
    ROM_FE,
    ROM_F8,
    ROM_F6,
    ROM_NOT_SUPPORTED,
    _ROM_MAX
};

// Define actions
enum Action : uint8_t
{
    ACTION_NOOP  = 0,
    ACTION_RIGHT = 1 << 0,
    ACTION_LEFT  = 1 << 1,
    ACTION_DOWN  = 1 << 2,
    ACTION_UP    = 1 << 3,
    ACTION_FIRE  = 1 << 4,
    ACTION_RESET = 1 << 5,

    ACTION_UPRIGHT        = ACTION_RIGHT | ACTION_UP,
    ACTION_UPLEFT         = ACTION_LEFT  | ACTION_UP,
    ACTION_DOWNRIGHT      = ACTION_RIGHT | ACTION_DOWN,
    ACTION_DOWNLEFT       = ACTION_LEFT  | ACTION_DOWN,
    ACTION_UPFIRE         = ACTION_FIRE  | ACTION_UP,
    ACTION_RIGHTFIRE      = ACTION_FIRE  | ACTION_RIGHT,
    ACTION_LEFTFIRE       = ACTION_FIRE  | ACTION_LEFT,
    ACTION_DOWNFIRE       = ACTION_FIRE  | ACTION_DOWN,
    ACTION_UPRIGHTFIRE    = ACTION_UP    | ACTION_RIGHTFIRE,
    ACTION_UPLEFTFIRE     = ACTION_UP    | ACTION_LEFTFIRE,
    ACTION_DOWNRIGHTFIRE  = ACTION_DOWN  | ACTION_RIGHTFIRE,
    ACTION_DOWNLEFTFIRE   = ACTION_DOWN  | ACTION_LEFTFIRE,

    _ACTION_MAX = 19,
};

enum HARWARE_CONFIG
{
    // hardware configuration
    NTSC_SCREEN_HEIGHT=210,
    PAL_SCREEN_HEIGHT=250,
    SCREEN_XOFFSET=0,
    SCREEN_WIDTH=160,
    SCREEN_YOFFSET=0,

    NTSC_SCREEN_SIZE=NTSC_SCREEN_HEIGHT * SCREEN_WIDTH,
    PAL_SCREEN_SIZE=PAL_SCREEN_HEIGHT * SCREEN_WIDTH,
    SCALED_SCREEN_SIZE=84 * 84,
    SCREEN_VISIBLE_LINES=192,

    SCANLINE_CPU_CYCLES=76,
    SCANLINE_COLOR_CYCLES=3*SCANLINE_CPU_CYCLES,

    HBLANK=68
};

// errors
enum EMUERROR
{
    INVALID_ROM=1,
    INVALID_MEMORY_ACCESS,
    INVALID_INSTRUCTION,
    ILLEGAL_OPERATION
};

enum EMUERRORSUBTYPE
{
    // INVALID_ROM
    INVALID_FILE_SIGNATURE,
    INVALID_ROM_CONFIG,
    UNEXPECTED_END_OFLAG_FILE,
    UNSUPPORTED_MAPPER_TYPE,

    // INVALID_MEMORY_ACCESS
    MAPPER_FAILURE,
    ADDRESS_OUT_OFLAG_RANGE,
    ILLEGAL_ADDRESS_WARP,
    MEMORY_NOT_EXECUTABLE,
    MEMORY_CANT_BE_READ,
    MEMORY_CANT_BE_WRITTEN,
    MEMORY_CANT_BE_COPIED,

    // INVALID_INSTRUCTION
    INVALID_OPCODE,
    INVALID_ADDRESS_MODE,

    // ILLEGAL_OPERATION
    IRQ_ALREADY_PENDING
};

enum
{
    ENV_UPDATE_SIZE  = 3500,
    ENV_NOOP_FRAMES  = 60,
    ENV_RESET_FRAMES = 4,
    ENV_BASE_FRAMES  = ENV_NOOP_FRAMES + ENV_RESET_FRAMES,
};

// status flags
enum SYS_FLAGS : uint32_t
{
    FLAG_CARRY            = 1 << 0,
    FLAG_ZERO             = 1 << 1,
    FLAG_INTERRUPT_OFF    = 1 << 2,
    FLAG_BCD              = 1 << 3,
    FLAG_BREAK            = 1 << 4,
    FLAG_RESERVED         = 1 << 5, // not used (always set)
    FLAG_OVERFLOW         = 1 << 6,
    FLAG_NEGATIVE         = 1 << 7,

    FLAG_DECIMAL          = FLAG_BCD,
    FLAG_SIGN             = FLAG_NEGATIVE,
    FLAG__NV              = FLAG_NEGATIVE|FLAG_OVERFLOW,

    FLAG_CON_RIGHT        = 1 << 8,
    FLAG_CON_LEFT         = 1 << 9,
    FLAG_CON_DOWN         = 1 << 10,
    FLAG_CON_UP           = 1 << 11,
    FLAG_CON_FIRE         = 1 << 12,
    FLAG_CON_RESET        = 1 << 13,
    FLAG_CON_PADDLES      = 1 << 14,
    FLAG_CON_SWAP         = 1 << 15,

    FLAG_SW_RESET_OFF     = 1 << 16,
    FLAG_SW_SELECT_OFF    = 1 << 17,
    FLAG_SW_UNUSED1       = 1 << 18,
    FLAG_SW_COLOR         = 1 << 19,
    FLAG_SW_UNUSED2       = 1 << 20,
    FLAG_SW_UNUSED3       = 1 << 21,
    FLAG_SW_LEFT_DIFFLAG_A   = 1 << 22,
    FLAG_SW_RIGHT_DIFFLAG_A  = 1 << 23,

    FLAG_CPU_HALT         = 1 << 24,
    FLAG_CPU_LAST_READ    = 1 << 25,
    FLAG_CPU_WRITE_BACK   = 1 << 26,
    FLAG_CPU_ERROR        = 1 << 27,
    FLAG_INT_NMI          = 1 << 28,
    FLAG_INT_BRK          = 1 << 29,
    FLAG_INT_IRQ          = 1 << 30,
    FLAG_INT_RST          = uint32_t(1 << 31),
};

enum : uint32_t
{
    FIELD_SYS_PS        = 0x000000FF,
    FIELD_SYS_SW        = 0x00FF0000,
    FIELD_SYS_SW_NODIFF = 0x003F0000,
    FIELD_SYS_INT       = 0xF0000000,
    FIELD_SYS_CON       = 0x00001F00,
    FIELD_SYS_CON_RESET = 0x00003F00,
};

enum TIA_FLAGS
{
    FLAG_TIA_CTRLPF       = 1 << 0,
    FLAG_TIA_DENABL       = 1 << 1,
    FLAG_TIA_DUMP         = 1 << 2,
    FLAG_TIA_ENAM0        = 1 << 3,
    FLAG_TIA_ENAM1        = 1 << 4,
    FLAG_TIA_ENABL        = 1 << 5,
    FLAG_TIA_HMOVE_ENABLE = 1 << 6,
    FLAG_TIA_HMOVE_ALLOW  = 1 << 7,

    FLAG_TIA_REFP0        = 1 << 8,
    FLAG_TIA_REFP1        = 1 << 9,
    FLAG_TIA_RESMP0       = 1 << 10,
    FLAG_TIA_RESMP1       = 1 << 11,
    FLAG_TIA_VBLANK1      = 1 << 12,
    FLAG_TIA_VBLANK2      = 1 << 13,
    FLAG_TIA_VDELP0       = 1 << 14,
    FLAG_TIA_VDELP1       = 1 << 15,

    FLAG_TIA_VDELBL       = 1 << 16,
    FLAG_TIA_PARTIAL      = 1 << 17,
    FLAG_TIA_IS_NTSC      = 1 << 18,
    FLAG_TIA_P0Bit        = 1 << 19,
    FLAG_TIA_M0Bit        = 1 << 20,
    FLAG_TIA_P1Bit        = 1 << 21,
    FLAG_TIA_M1Bit        = 1 << 22,
    FLAG_TIA_BLBit        = 1 << 23,

    FLAG_TIA_PFBit        = 1 << 24,
    FLAG_RIOT_READ_INT    = 1 << 25,
    FLAG_ALE_STARTED      = 1 << 26,
    FLAG_ALE_TERMINAL     = 1 << 27,
    FLAG_ALE_LOST_LIFE    = 1 << 28,
};

enum : uint32_t
{
    FIELD_TIA_STATUS   = 0x01FBFF7F,
    FIELD_TIA_ENABLED  = 0x01F80000,
    FIELD_FRAME_NUMBER = 0x0000FFFF,
    FIELD_START_ACTION = 0x00FF0000,
    FIELD_START_NUMBER = 0xFF000000
};

enum Control_Jack : uint8_t
{
    Control_Left,
    Control_Right
};

enum Type : uint8_t
{
    Paddles,
    Joystick,
};

enum Control_DigitalPin : uint8_t
{
    Control_One,
    Control_Two,
    Control_Three,
    Control_Four,
    Control_Six
};

enum Control_AnalogPin : uint8_t
{
    Control_Five,
    Control_Nine
};

enum : int32_t
{
    /// Constant which represents maximum resistance for analog pins
    Control_maximumResistance = 0x7FFFFFFF,
    /// Constant which represents minimum resistance for analog pins
    Control_minimumResistance = 0x00000000
};

// M6532 (RIOT) Write/Read register names
enum M6532Register
{
    ADR_SWCHA  = 0x280,
    ADR_SWACNT = 0x281,
    ADR_SWCHB  = 0x282,
    ADR_SWBCNT = 0x283,
    ADR_INTIM  = 0x284,

    ADR_TIM1T  = 0x294,
    ADR_TIM8T  = 0x295,
    ADR_TIM64T = 0x296,
    ADR_T1024T = 0x297,
};

enum : uint32_t
{
    FIELD_RIOT_TIMER = 0x000000FF,
    FIELD_RIOT_SHIFT = 0x0000FF00,
    FIELD_RIOT_DDRA  = 0x00FF0000,
    FIELD_RIOT_GAME  = 0xFF000000,
};

enum TIABit
{
    P0Bit       = 0x01,  // Bit for Player 0
    M0Bit       = 0x02,  // Bit for Missle 0
    P1Bit       = 0x04,  // Bit for Player 1
    M1Bit       = 0x08,  // Bit for Missle 1
    BLBit       = 0x10,  // Bit for Ball
    PFBit       = 0x20,  // Bit for Playfield
    ScoreBit    = 0x40,  // Bit for Playfield score mode
    PriorityBit = 0x80   // Bit for Playfield priority
};

enum TIAColor
{
    BKColor     = 0,  // Color index for Background
    PFColor     = 1,  // Color index for Playfield
    P0Color     = 2,  // Color index for Player 0
    P1Color     = 3,  // Color index for Player 1
    M0Color     = 4,  // Color index for Missle 0
    M1Color     = 5,  // Color index for Missle 1
    BLColor     = 6,  // Color index for Ball
    HBLANKColor = 7   // Color index for HMove blank area
};

enum CollisionBit
{
    Cx_M0P1 = 0x0001, // Missle0 - Player1   collision
    Cx_M0P0 = 0x0002, // Missle0 - Player0   collision
    Cx_M1P0 = 0x0004, // Missle1 - Player0   collision
    Cx_M1P1 = 0x0008, // Missle1 - Player1   collision
    Cx_P0PF = 0x0010, // Player0 - Playfield collision
    Cx_P0BL = 0x0020, // Player0 - Ball      collision
    Cx_P1PF = 0x0040, // Player1 - Playfield collision
    Cx_P1BL = 0x0080, // Player1 - Ball      collision
    Cx_M0PF = 0x0100, // Missle0 - Playfield collision
    Cx_M0BL = 0x0200, // Missle0 - Ball      collision
    Cx_M1PF = 0x0400, // Missle1 - Playfield collision
    Cx_M1BL = 0x0800, // Missle1 - Ball      collision
    Cx_BLPF = 0x1000, // Ball - Playfield    collision
    Cx_P0P1 = 0x2000, // Player0 - Player1   collision
    Cx_M0M1 = 0x4000  // Missle0 - Missle1   collision
};

// TIA Write/Read register names
enum TIARegister
{
    ADR_VSYNC   = 0x00,  // Write: vertical sync set-clear (D1)
    ADR_VBLANK  = 0x01,  // Write: vertical blank set-clear (D7-6,D1)
    ADR_WSYNC   = 0x02,  // Write: wait for leading edge of hrz. blank (strobe)
    ADR_RSYNC   = 0x03,  // Write: reset hrz. sync counter (strobe)
    ADR_NUSIZ0  = 0x04,  // Write: number-size player-missle 0 (D5-0)
    ADR_NUSIZ1  = 0x05,  // Write: number-size player-missle 1 (D5-0)
    ADR_COLUP0  = 0x06,  // Write: color-lum player 0 (D7-1)
    ADR_COLUP1  = 0x07,  // Write: color-lum player 1 (D7-1)
    ADR_COLUPF  = 0x08,  // Write: color-lum playfield (D7-1)
    ADR_COLUBK  = 0x09,  // Write: color-lum background (D7-1)
    ADR_CTRLPF  = 0x0a,  // Write: cntrl playfield ballsize & coll. (D5-4,D2-0)
    ADR_REFP0   = 0x0b,  // Write: reflect player 0 (D3)
    ADR_REFP1   = 0x0c,  // Write: reflect player 1 (D3)
    ADR_PF0     = 0x0d,  // Write: playfield register byte 0 (D7-4)
    ADR_PF1     = 0x0e,  // Write: playfield register byte 1 (D7-0)
    ADR_PF2     = 0x0f,  // Write: playfield register byte 2 (D7-0)
    ADR_RESP0   = 0x10,  // Write: reset player 0 (strobe)
    ADR_RESP1   = 0x11,  // Write: reset player 1 (strobe)
    ADR_RESM0   = 0x12,  // Write: reset missle 0 (strobe)
    ADR_RESM1   = 0x13,  // Write: reset missle 1 (strobe)
    ADR_RESBL   = 0x14,  // Write: reset ball (strobe)
    ADR_AUDC0   = 0x15,  // Write: audio control 0 (D3-0)
    ADR_AUDC1   = 0x16,  // Write: audio control 1 (D4-0)
    ADR_AUDF0   = 0x17,  // Write: audio frequency 0 (D4-0)
    ADR_AUDF1   = 0x18,  // Write: audio frequency 1 (D3-0)
    ADR_AUDV0   = 0x19,  // Write: audio volume 0 (D3-0)
    ADR_AUDV1   = 0x1a,  // Write: audio volume 1 (D3-0)
    ADR_GRP0    = 0x1b,  // Write: graphics player 0 (D7-0)
    ADR_GRP1    = 0x1c,  // Write: graphics player 1 (D7-0)
    ADR_ENAM0   = 0x1d,  // Write: graphics (enable) missle 0 (D1)
    ADR_ENAM1   = 0x1e,  // Write: graphics (enable) missle 1 (D1)
    ADR_ENABL   = 0x1f,  // Write: graphics (enable) ball (D1)
    ADR_HMP0    = 0x20,  // Write: horizontal motion player 0 (D7-4)
    ADR_HMP1    = 0x21,  // Write: horizontal motion player 1 (D7-4)
    ADR_HMM0    = 0x22,  // Write: horizontal motion missle 0 (D7-4)
    ADR_HMM1    = 0x23,  // Write: horizontal motion missle 1 (D7-4)
    ADR_HMBL    = 0x24,  // Write: horizontal motion ball (D7-4)
    ADR_VDELP0  = 0x25,  // Write: vertical delay player 0 (D0)
    ADR_VDELP1  = 0x26,  // Write: vertical delay player 1 (D0)
    ADR_VDELBL  = 0x27,  // Write: vertical delay ball (D0)
    ADR_RESMP0  = 0x28,  // Write: reset missle 0 to player 0 (D1)
    ADR_RESMP1  = 0x29,  // Write: reset missle 1 to player 1 (D1)
    ADR_HMOVE   = 0x2a,  // Write: apply horizontal motion (strobe)
    ADR_HMCLR   = 0x2b,  // Write: clear horizontal motion registers (strobe)
    ADR_CXCLR   = 0x2c,  // Write: clear collision latches (strobe)
};

enum
{
    ADR_CXM0P   = 0x0,  // Read collision: D7=(M0,P1); D6=(M0,P0)
    ADR_CXM1P   = 0x1,  // Read collision: D7=(M1,P0); D6=(M1,P1)
    ADR_CXP0FB  = 0x2,  // Read collision: D7=(P0,PF); D6=(P0,BL)
    ADR_CXP1FB  = 0x3,  // Read collision: D7=(P1,PF); D6=(P1,BL)
    ADR_CXM0FB  = 0x4,  // Read collision: D7=(M0,PF); D6=(M0,BL)
    ADR_CXM1FB  = 0x5,  // Read collision: D7=(M1,PF); D6=(M1,BL)
    ADR_CXBLPF  = 0x6,  // Read collision: D7=(BL,PF); D6=(unused)
    ADR_CXPPMM  = 0x7,  // Read collision: D7=(P0,P1); D6=(M0,M1)
    ADR_INPT0   = 0x8,  // Read pot port: D7
    ADR_INPT1   = 0x9,  // Read pot port: D7
    ADR_INPT2   = 0xa,  // Read pot port: D7
    ADR_INPT3   = 0xb,  // Read pot port: D7
    ADR_INPT4   = 0xc,  // Read P1 joystick trigger: D7
    ADR_INPT5   = 0xd   // Read P2 joystick trigger: D7
};

enum : uint32_t
{
    FIELD_GRP0  = 0x000000FF,
    FIELD_GRP1  = 0x0000FF00,
    FIELD_DGRP0 = 0x00FF0000,
    FIELD_DGRP1 = 0xFF000000,

    FIELD_POSP0 = 0x000000FF,
    FIELD_POSP1 = 0x0000FF00,
    FIELD_POSM0 = 0x00FF0000,
    FIELD_POSM1 = 0xFF000000,

    FIELD_PF0   = 0x0000000F,
    FIELD_PF1   = 0x00000FF0,
    FIELD_PF2   = 0x000FF000,
    FIELD_PFALL = 0x000FFFFF,
    FIELD_NUSIZ0_MODE = 0x00700000,
    FIELD_NUSIZ1_MODE = 0x03800000,
    FIELD_NUSIZ0_SIZE = 0x0C000000,
    FIELD_NUSIZ1_SIZE = 0x30000000,
    FIELD_CTRLPF = 0xC0000000,

    FIELD_HMP0  = 0x0000000F,
    FIELD_HMP1  = 0x000000F0,
    FIELD_HMM0  = 0x00000F00,
    FIELD_HMM1  = 0x0000F000,
    FIELD_HMBL  = 0x000F0000,
    FIELD_HMALL = 0x000FFFFF,
    FIELD_POSBL = 0x0FF00000,

    FIELD_COLUBK = 0x000000FF,
    FIELD_COLUPF = 0x0000FF00,
    FIELD_COLUP0 = 0x00FF0000,
    FIELD_COLUP1 = 0xFF000000,
};

} // end namespace atari
} // end namespace cule


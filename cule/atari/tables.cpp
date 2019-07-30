#pragma once

#include <cule/config.hpp>

#include <cule/atari/tables.hpp>

#include <cassert>
#include <cstdlib>

namespace cule
{
namespace atari
{

#define hmove_accessor(x) CULE_ARRAY_ACCESSOR(ourHMOVEBlankEnableCycles[x])
#define poke_accessor(x) CULE_ARRAY_ACCESSOR(ourPokeDelayTable[x])
#define motion_accessor(x, y) CULE_ARRAY_ACCESSOR(ourCompleteMotionTable[x][y])

#define player_position_accessor(x, y, z) CULE_ARRAY_ACCESSOR(ourPlayerPositionResetWhenTable[x][y][z])
#define ball_accessor(x, y, z) CULE_ARRAY_ACCESSOR(ourBallMaskTable[x][y][z])
#define disabled_accessor(x) CULE_ARRAY_ACCESSOR(ourDisabledMaskTable[x])
#define player_mask_accessor(x, y, z, w) CULE_ARRAY_ACCESSOR(ourPlayerMaskTable[x][y][z][w])
#define player_reflect_accessor(x) CULE_ARRAY_ACCESSOR(ourPlayerReflectTable[x])
#define missle_accessor(x, y, z, w) CULE_ARRAY_ACCESSOR(ourMissleMaskTable[x][y][z][w])
#define priority_accessor(x, y) CULE_ARRAY_ACCESSOR(ourPriorityEncoder[x][y])
#define collision_accessor(x) CULE_ARRAY_ACCESSOR(ourCollisionTable[x])
#define playfield_accessor(x, y) CULE_ARRAY_ACCESSOR(ourPlayfieldTable[x][y])

int8_t   ourPlayerPositionResetWhenTable[8][160][160];
uint8_t  ourBallMaskTable[4][4][320];
uint8_t  ourDisabledMaskTable[640];
uint8_t  ourPlayerMaskTable[4][2][8][320];
uint8_t  ourPlayerReflectTable[256];
uint8_t  ourMissleMaskTable[4][8][4][320];
uint8_t  ourPriorityEncoder[2][256];
uint16_t ourCollisionTable[64];
uint32_t ourPlayfieldTable[2][160];

const int16_t ourPokeDelayTable[64] =
{
    0,  1,  0,  0,  8,  8,  0,  0,  0,  0,  0,  1,  1, -1, -1, -1,
    0,  0,  8,  8,  0,  0,  0,  0,  0,  0,  0,  1,  1,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
};

const bool ourHMOVEBlankEnableCycles[76] =
{
    true,  true,  true,  true,  true,  true,  true,  true,  true,  true,   // 00
    true,  true,  true,  true,  true,  true,  true,  true,  true,  true,   // 10
    true,  false, false, false, false, false, false, false, false, false,  // 20
    false, false, false, false, false, false, false, false, false, false,  // 30
    false, false, false, false, false, false, false, false, false, false,  // 40
    false, false, false, false, false, false, false, false, false, false,  // 50
    false, false, false, false, false, false, false, false, false, false,  // 60
    false, false, false, false, false, true                                // 70
};

const int16_t ourCompleteMotionTable[76][16] =
{
    { 0, -1, -2, -3, -4, -5, -6, -7,  8,  7,  6,  5,  4,  3,  2,  1}, // HBLANK
    { 0, -1, -2, -3, -4, -5, -6, -7,  8,  7,  6,  5,  4,  3,  2,  1}, // HBLANK
    { 0, -1, -2, -3, -4, -5, -6, -7,  8,  7,  6,  5,  4,  3,  2,  1}, // HBLANK
    { 0, -1, -2, -3, -4, -5, -6, -7,  8,  7,  6,  5,  4,  3,  2,  1}, // HBLANK
    { 0, -1, -2, -3, -4, -5, -6, -6,  8,  7,  6,  5,  4,  3,  2,  1}, // HBLANK
    { 0, -1, -2, -3, -4, -5, -5, -5,  8,  7,  6,  5,  4,  3,  2,  1}, // HBLANK
    { 0, -1, -2, -3, -4, -5, -5, -5,  8,  7,  6,  5,  4,  3,  2,  1}, // HBLANK
    { 0, -1, -2, -3, -4, -4, -4, -4,  8,  7,  6,  5,  4,  3,  2,  1}, // HBLANK
    { 0, -1, -2, -3, -3, -3, -3, -3,  8,  7,  6,  5,  4,  3,  2,  1}, // HBLANK
    { 0, -1, -2, -2, -2, -2, -2, -2,  8,  7,  6,  5,  4,  3,  2,  1}, // HBLANK
    { 0, -1, -2, -2, -2, -2, -2, -2,  8,  7,  6,  5,  4,  3,  2,  1}, // HBLANK
    { 0, -1, -1, -1, -1, -1, -1, -1,  8,  7,  6,  5,  4,  3,  2,  1}, // HBLANK
    { 0,  0,  0,  0,  0,  0,  0,  0,  8,  7,  6,  5,  4,  3,  2,  1}, // HBLANK
    { 1,  1,  1,  1,  1,  1,  1,  1,  8,  7,  6,  5,  4,  3,  2,  1}, // HBLANK
    { 1,  1,  1,  1,  1,  1,  1,  1,  8,  7,  6,  5,  4,  3,  2,  1}, // HBLANK
    { 2,  2,  2,  2,  2,  2,  2,  2,  8,  7,  6,  5,  4,  3,  2,  2}, // HBLANK
    { 3,  3,  3,  3,  3,  3,  3,  3,  8,  7,  6,  5,  4,  3,  3,  3}, // HBLANK
    { 4,  4,  4,  4,  4,  4,  4,  4,  8,  7,  6,  5,  4,  4,  4,  4}, // HBLANK
    { 4,  4,  4,  4,  4,  4,  4,  4,  8,  7,  6,  5,  4,  4,  4,  4}, // HBLANK
    { 5,  5,  5,  5,  5,  5,  5,  5,  8,  7,  6,  5,  5,  5,  5,  5}, // HBLANK
    { 6,  6,  6,  6,  6,  6,  6,  6,  8,  7,  6,  6,  6,  6,  6,  6}, // HBLANK
    { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    { 0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0},
    { 0,  0,  0,  0,  0,  0, -1, -2,  0,  0,  0,  0,  0,  0,  0,  0},
    { 0,  0,  0,  0,  0, -1, -2, -3,  0,  0,  0,  0,  0,  0,  0,  0},
    { 0,  0,  0,  0,  0, -1, -2, -3,  0,  0,  0,  0,  0,  0,  0,  0},
    { 0,  0,  0,  0, -1, -2, -3, -4,  0,  0,  0,  0,  0,  0,  0,  0},
    { 0,  0,  0, -1, -2, -3, -4, -5,  0,  0,  0,  0,  0,  0,  0,  0},
    { 0,  0, -1, -2, -3, -4, -5, -6,  0,  0,  0,  0,  0,  0,  0,  0},
    { 0,  0, -1, -2, -3, -4, -5, -6,  0,  0,  0,  0,  0,  0,  0,  0},
    { 0, -1, -2, -3, -4, -5, -6, -7,  0,  0,  0,  0,  0,  0,  0,  0},
    {-1, -2, -3, -4, -5, -6, -7, -8,  0,  0,  0,  0,  0,  0,  0,  0},
    {-2, -3, -4, -5, -6, -7, -8, -9,  0,  0,  0,  0,  0,  0,  0, -1},
    {-2, -3, -4, -5, -6, -7, -8, -9,  0,  0,  0,  0,  0,  0,  0, -1},
    {-3, -4, -5, -6, -7, -8, -9,-10,  0,  0,  0,  0,  0,  0, -1, -2},
    {-4, -5, -6, -7, -8, -9,-10,-11,  0,  0,  0,  0,  0, -1, -2, -3},
    {-5, -6, -7, -8, -9,-10,-11,-12,  0,  0,  0,  0, -1, -2, -3, -4},
    {-5, -6, -7, -8, -9,-10,-11,-12,  0,  0,  0,  0, -1, -2, -3, -4},
    {-6, -7, -8, -9,-10,-11,-12,-13,  0,  0,  0, -1, -2, -3, -4, -5},
    {-7, -8, -9,-10,-11,-12,-13,-14,  0,  0, -1, -2, -3, -4, -5, -6},
    {-8, -9,-10,-11,-12,-13,-14,-15,  0, -1, -2, -3, -4, -5, -6, -7},
    {-8, -9,-10,-11,-12,-13,-14,-15,  0, -1, -2, -3, -4, -5, -6, -7},
    { 0, -1, -2, -3, -4, -5, -6, -7,  8,  7,  6,  5,  4,  3,  2,  1}  // HBLANK
};

// Compute the ball mask table
void computeBallMaskTable()
{
    // First, calculate masks for alignment 0
    for(int32_t size = 0; size < 4; ++size)
    {
        int32_t x;

        // Set all of the masks to false to start with
        for(x = 0; x < 160; ++x)
        {
            ourBallMaskTable[0][size][x] = false;
        }

        // Set the necessary fields true
        for(x = 0; x < 160 + 8; ++x)
        {
            if((x >= 0) && (x < (1 << size)))
            {
                ourBallMaskTable[0][size][x % 160] = true;
            }
        }

        // Copy fields into the wrap-around area of the mask
        for(x = 0; x < 160; ++x)
        {
            ourBallMaskTable[0][size][x + 160] = ourBallMaskTable[0][size][x];
        }
    }

    // Now, copy data for alignments of 1, 2 and 3
    for(uint32_t align = 1; align < 4; ++align)
    {
        for(uint32_t size = 0; size < 4; ++size)
        {
            for(uint32_t x = 0; x < 320; ++x)
            {
                ourBallMaskTable[align][size][x] =
                    ourBallMaskTable[0][size][(x + 320 - align) % 320];
            }
        }
    }
}

// Compute the collision decode table
void computeCollisionTable()
{
    for(uint8_t i = 0; i < 64; ++i)
    {
        ourCollisionTable[i] = 0;

        if((i & M0Bit) && (i & P1Bit))    // M0-P1
            ourCollisionTable[i] |= 0x0001;

        if((i & M0Bit) && (i & P0Bit))    // M0-P0
            ourCollisionTable[i] |= 0x0002;

        if((i & M1Bit) && (i & P0Bit))    // M1-P0
            ourCollisionTable[i] |= 0x0004;

        if((i & M1Bit) && (i & P1Bit))    // M1-P1
            ourCollisionTable[i] |= 0x0008;

        if((i & P0Bit) && (i & PFBit))    // P0-PF
            ourCollisionTable[i] |= 0x0010;

        if((i & P0Bit) && (i & BLBit))    // P0-BL
            ourCollisionTable[i] |= 0x0020;

        if((i & P1Bit) && (i & PFBit))    // P1-PF
            ourCollisionTable[i] |= 0x0040;

        if((i & P1Bit) && (i & BLBit))    // P1-BL
            ourCollisionTable[i] |= 0x0080;

        if((i & M0Bit) && (i & PFBit))    // M0-PF
            ourCollisionTable[i] |= 0x0100;

        if((i & M0Bit) && (i & BLBit))    // M0-BL
            ourCollisionTable[i] |= 0x0200;

        if((i & M1Bit) && (i & PFBit))    // M1-PF
            ourCollisionTable[i] |= 0x0400;

        if((i & M1Bit) && (i & BLBit))    // M1-BL
            ourCollisionTable[i] |= 0x0800;

        if((i & BLBit) && (i & PFBit))    // BL-PF
            ourCollisionTable[i] |= 0x1000;

        if((i & P0Bit) && (i & P1Bit))    // P0-P1
            ourCollisionTable[i] |= 0x2000;

        if((i & M0Bit) && (i & M1Bit))    // M0-M1
            ourCollisionTable[i] |= 0x4000;
    }
}

// Compute the missle mask table
void computeMissleMaskTable()
{
    // First, calculate masks for alignment 0
    int32_t x, size, number;

    // Clear the missle table to start with
    for(number = 0; number < 8; ++number)
        for(size = 0; size < 4; ++size)
            for(x = 0; x < 160; ++x)
                ourMissleMaskTable[0][number][size][x] = false;

    for(number = 0; number < 8; ++number)
    {
        for(size = 0; size < 4; ++size)
        {
            for(x = 0; x < 160 + 72; ++x)
            {
                // Only one copy of the missle
                if((number == 0x00) || (number == 0x05) || (number == 0x07))
                {
                    if((x >= 0) && (x < (1 << size)))
                        ourMissleMaskTable[0][number][size][x % 160] = true;
                }
                // Two copies - close
                else if(number == 0x01)
                {
                    if((x >= 0) && (x < (1 << size)))
                        ourMissleMaskTable[0][number][size][x % 160] = true;
                    else if(((x - 16) >= 0) && ((x - 16) < (1 << size)))
                        ourMissleMaskTable[0][number][size][x % 160] = true;
                }
                // Two copies - medium
                else if(number == 0x02)
                {
                    if((x >= 0) && (x < (1 << size)))
                        ourMissleMaskTable[0][number][size][x % 160] = true;
                    else if(((x - 32) >= 0) && ((x - 32) < (1 << size)))
                        ourMissleMaskTable[0][number][size][x % 160] = true;
                }
                // Three copies - close
                else if(number == 0x03)
                {
                    if((x >= 0) && (x < (1 << size)))
                        ourMissleMaskTable[0][number][size][x % 160] = true;
                    else if(((x - 16) >= 0) && ((x - 16) < (1 << size)))
                        ourMissleMaskTable[0][number][size][x % 160] = true;
                    else if(((x - 32) >= 0) && ((x - 32) < (1 << size)))
                        ourMissleMaskTable[0][number][size][x % 160] = true;
                }
                // Two copies - wide
                else if(number == 0x04)
                {
                    if((x >= 0) && (x < (1 << size)))
                        ourMissleMaskTable[0][number][size][x % 160] = true;
                    else if(((x - 64) >= 0) && ((x - 64) < (1 << size)))
                        ourMissleMaskTable[0][number][size][x % 160] = true;
                }
                // Three copies - medium
                else if(number == 0x06)
                {
                    if((x >= 0) && (x < (1 << size)))
                        ourMissleMaskTable[0][number][size][x % 160] = true;
                    else if(((x - 32) >= 0) && ((x - 32) < (1 << size)))
                        ourMissleMaskTable[0][number][size][x % 160] = true;
                    else if(((x - 64) >= 0) && ((x - 64) < (1 << size)))
                        ourMissleMaskTable[0][number][size][x % 160] = true;
                }
            }

            // Copy data into wrap-around area
            for(x = 0; x < 160; ++x)
                ourMissleMaskTable[0][number][size][x + 160] =
                    ourMissleMaskTable[0][number][size][x];
        }
    }

    // Now, copy data for alignments of 1, 2 and 3
    for(int32_t align = 1; align < 4; ++align)
    {
        for(number = 0; number < 8; ++number)
        {
            for(size = 0; size < 4; ++size)
            {
                for(x = 0; x < 320; ++x)
                {
                    ourMissleMaskTable[align][number][size][x] =
                        ourMissleMaskTable[0][number][size][(x + 320 - align) % 320];
                }
            }
        }
    }
}

// Compute the player mask table
void computePlayerMaskTable()
{
    // First, calculate masks for alignment 0
    int32_t x, enable, mode;

    // Set the player mask table to all zeros
    for(enable = 0; enable < 2; ++enable)
        for(mode = 0; mode < 8; ++mode)
            for(x = 0; x < 160; ++x)
                ourPlayerMaskTable[0][enable][mode][x] = 0x00;

    // Now, compute the player mask table
    for(enable = 0; enable < 2; ++enable)
    {
        for(mode = 0; mode < 8; ++mode)
        {
            for(x = 0; x < 160 + 72; ++x)
            {
                if(mode == 0x00)
                {
                    if((enable == 0) && (x >= 0) && (x < 8))
                        ourPlayerMaskTable[0][enable][mode][x % 160] = 0x80 >> x;
                }
                else if(mode == 0x01)
                {
                    if((enable == 0) && (x >= 0) && (x < 8))
                        ourPlayerMaskTable[0][enable][mode][x % 160] = 0x80 >> x;
                    else if(((x - 16) >= 0) && ((x - 16) < 8))
                        ourPlayerMaskTable[0][enable][mode][x % 160] = 0x80 >> (x - 16);
                }
                else if(mode == 0x02)
                {
                    if((enable == 0) && (x >= 0) && (x < 8))
                        ourPlayerMaskTable[0][enable][mode][x % 160] = 0x80 >> x;
                    else if(((x - 32) >= 0) && ((x - 32) < 8))
                        ourPlayerMaskTable[0][enable][mode][x % 160] = 0x80 >> (x - 32);
                }
                else if(mode == 0x03)
                {
                    if((enable == 0) && (x >= 0) && (x < 8))
                        ourPlayerMaskTable[0][enable][mode][x % 160] = 0x80 >> x;
                    else if(((x - 16) >= 0) && ((x - 16) < 8))
                        ourPlayerMaskTable[0][enable][mode][x % 160] = 0x80 >> (x - 16);
                    else if(((x - 32) >= 0) && ((x - 32) < 8))
                        ourPlayerMaskTable[0][enable][mode][x % 160] = 0x80 >> (x - 32);
                }
                else if(mode == 0x04)
                {
                    if((enable == 0) && (x >= 0) && (x < 8))
                        ourPlayerMaskTable[0][enable][mode][x % 160] = 0x80 >> x;
                    else if(((x - 64) >= 0) && ((x - 64) < 8))
                        ourPlayerMaskTable[0][enable][mode][x % 160] = 0x80 >> (x - 64);
                }
                else if(mode == 0x05)
                {
                    // For some reason in double size mode the player's output
                    // is delayed by one pixel thus we use > instead of >=
                    if((enable == 0) && (x > 0) && (x <= 16))
                        ourPlayerMaskTable[0][enable][mode][x % 160] = 0x80 >> ((x - 1)/2);
                }
                else if(mode == 0x06)
                {
                    if((enable == 0) && (x >= 0) && (x < 8))
                        ourPlayerMaskTable[0][enable][mode][x % 160] = 0x80 >> x;
                    else if(((x - 32) >= 0) && ((x - 32) < 8))
                        ourPlayerMaskTable[0][enable][mode][x % 160] = 0x80 >> (x - 32);
                    else if(((x - 64) >= 0) && ((x - 64) < 8))
                        ourPlayerMaskTable[0][enable][mode][x % 160] = 0x80 >> (x - 64);
                }
                else if(mode == 0x07)
                {
                    // For some reason in quad size mode the player's output
                    // is delayed by one pixel thus we use > instead of >=
                    if((enable == 0) && (x > 0) && (x <= 32))
                        ourPlayerMaskTable[0][enable][mode][x % 160] = 0x80 >> ((x - 1)/4);
                }
            }

            // Copy data into wrap-around area
            for(x = 0; x < 160; ++x)
            {
                ourPlayerMaskTable[0][enable][mode][x + 160] =
                    ourPlayerMaskTable[0][enable][mode][x];
            }
        }
    }

    // Now, copy data for alignments of 1, 2 and 3
    for(int32_t align = 1; align < 4; ++align)
    {
        for(enable = 0; enable < 2; ++enable)
        {
            for(mode = 0; mode < 8; ++mode)
            {
                for(x = 0; x < 320; ++x)
                {
                    ourPlayerMaskTable[align][enable][mode][x] =
                        ourPlayerMaskTable[0][enable][mode][(x + 320 - align) % 320];
                }
            }
        }
    }
}

// Compute the player position reset when table
void computePlayerPositionResetWhenTable()
{
    uint32_t mode, oldx, newx;

    // Loop through all player modes, all old player positions, and all new
    // player positions and determine where the new position is located:
    // 1 means the new position is within the display of an old copy of the
    // player, -1 means the new position is within the delay portion of an
    // old copy of the player, and 0 means it's neither of these two
    for(mode = 0; mode < 8; ++mode)
    {
        for(oldx = 0; oldx < 160; ++oldx)
        {
            // Set everything to 0 for non-delay/non-display section
            for(newx = 0; newx < 160; ++newx)
            {
                ourPlayerPositionResetWhenTable[mode][oldx][newx] = 0;
            }

            // Now, we'll set the entries for non-delay/non-display section
            for(newx = 0; newx < 160 + 72 + 5; ++newx)
            {
                if(mode == 0x00)
                {
                    if((newx >= oldx) && (newx < (oldx + 4)))
                        ourPlayerPositionResetWhenTable[mode][oldx][newx % 160] = -1;

                    if((newx >= oldx + 4) && (newx < (oldx + 4 + 8)))
                        ourPlayerPositionResetWhenTable[mode][oldx][newx % 160] = 1;
                }
                else if(mode == 0x01)
                {
                    if((newx >= oldx) && (newx < (oldx + 4)))
                        ourPlayerPositionResetWhenTable[mode][oldx][newx % 160] = -1;
                    else if((newx >= (oldx + 16)) && (newx < (oldx + 16 + 4)))
                        ourPlayerPositionResetWhenTable[mode][oldx][newx % 160] = -1;

                    if((newx >= oldx + 4) && (newx < (oldx + 4 + 8)))
                        ourPlayerPositionResetWhenTable[mode][oldx][newx % 160] = 1;
                    else if((newx >= oldx + 16 + 4) && (newx < (oldx + 16 + 4 + 8)))
                        ourPlayerPositionResetWhenTable[mode][oldx][newx % 160] = 1;
                }
                else if(mode == 0x02)
                {
                    if((newx >= oldx) && (newx < (oldx + 4)))
                        ourPlayerPositionResetWhenTable[mode][oldx][newx % 160] = -1;
                    else if((newx >= (oldx + 32)) && (newx < (oldx + 32 + 4)))
                        ourPlayerPositionResetWhenTable[mode][oldx][newx % 160] = -1;

                    if((newx >= oldx + 4) && (newx < (oldx + 4 + 8)))
                        ourPlayerPositionResetWhenTable[mode][oldx][newx % 160] = 1;
                    else if((newx >= oldx + 32 + 4) && (newx < (oldx + 32 + 4 + 8)))
                        ourPlayerPositionResetWhenTable[mode][oldx][newx % 160] = 1;
                }
                else if(mode == 0x03)
                {
                    if((newx >= oldx) && (newx < (oldx + 4)))
                        ourPlayerPositionResetWhenTable[mode][oldx][newx % 160] = -1;
                    else if((newx >= (oldx + 16)) && (newx < (oldx + 16 + 4)))
                        ourPlayerPositionResetWhenTable[mode][oldx][newx % 160] = -1;
                    else if((newx >= (oldx + 32)) && (newx < (oldx + 32 + 4)))
                        ourPlayerPositionResetWhenTable[mode][oldx][newx % 160] = -1;

                    if((newx >= oldx + 4) && (newx < (oldx + 4 + 8)))
                        ourPlayerPositionResetWhenTable[mode][oldx][newx % 160] = 1;
                    else if((newx >= oldx + 16 + 4) && (newx < (oldx + 16 + 4 + 8)))
                        ourPlayerPositionResetWhenTable[mode][oldx][newx % 160] = 1;
                    else if((newx >= oldx + 32 + 4) && (newx < (oldx + 32 + 4 + 8)))
                        ourPlayerPositionResetWhenTable[mode][oldx][newx % 160] = 1;
                }
                else if(mode == 0x04)
                {
                    if((newx >= oldx) && (newx < (oldx + 4)))
                        ourPlayerPositionResetWhenTable[mode][oldx][newx % 160] = -1;
                    else if((newx >= (oldx + 64)) && (newx < (oldx + 64 + 4)))
                        ourPlayerPositionResetWhenTable[mode][oldx][newx % 160] = -1;

                    if((newx >= oldx + 4) && (newx < (oldx + 4 + 8)))
                        ourPlayerPositionResetWhenTable[mode][oldx][newx % 160] = 1;
                    else if((newx >= oldx + 64 + 4) && (newx < (oldx + 64 + 4 + 8)))
                        ourPlayerPositionResetWhenTable[mode][oldx][newx % 160] = 1;
                }
                else if(mode == 0x05)
                {
                    if((newx >= oldx) && (newx < (oldx + 4)))
                        ourPlayerPositionResetWhenTable[mode][oldx][newx % 160] = -1;

                    if((newx >= oldx + 4) && (newx < (oldx + 4 + 16)))
                        ourPlayerPositionResetWhenTable[mode][oldx][newx % 160] = 1;
                }
                else if(mode == 0x06)
                {
                    if((newx >= oldx) && (newx < (oldx + 4)))
                        ourPlayerPositionResetWhenTable[mode][oldx][newx % 160] = -1;
                    else if((newx >= (oldx + 32)) && (newx < (oldx + 32 + 4)))
                        ourPlayerPositionResetWhenTable[mode][oldx][newx % 160] = -1;
                    else if((newx >= (oldx + 64)) && (newx < (oldx + 64 + 4)))
                        ourPlayerPositionResetWhenTable[mode][oldx][newx % 160] = -1;

                    if((newx >= oldx + 4) && (newx < (oldx + 4 + 8)))
                        ourPlayerPositionResetWhenTable[mode][oldx][newx % 160] = 1;
                    else if((newx >= oldx + 32 + 4) && (newx < (oldx + 32 + 4 + 8)))
                        ourPlayerPositionResetWhenTable[mode][oldx][newx % 160] = 1;
                    else if((newx >= oldx + 64 + 4) && (newx < (oldx + 64 + 4 + 8)))
                        ourPlayerPositionResetWhenTable[mode][oldx][newx % 160] = 1;
                }
                else if(mode == 0x07)
                {
                    if((newx >= oldx) && (newx < (oldx + 4)))
                        ourPlayerPositionResetWhenTable[mode][oldx][newx % 160] = -1;

                    if((newx >= oldx + 4) && (newx < (oldx + 4 + 32)))
                        ourPlayerPositionResetWhenTable[mode][oldx][newx % 160] = 1;
                }
            }

            // Let's do a sanity check on our table entries
            uint32_t s1 = 0, s2 = 0;
            for(newx = 0; newx < 160; ++newx)
            {
                if(ourPlayerPositionResetWhenTable[mode][oldx][newx] == -1)
                    ++s1;
                if(ourPlayerPositionResetWhenTable[mode][oldx][newx] == 1)
                    ++s2;
            }
            assert((s1 % 4 == 0) && (s2 % 8 == 0));
        }
    }
}

// Compute the player reflect table
void computePlayerReflectTable()
{
    for(uint16_t i = 0; i < 256; ++i)
    {
        uint8_t r = 0;

        for(uint16_t t = 1; t <= 128; t *= 2)
        {
            r = ((r << 1) | ((i & t)) ? uint8_t(1) : uint8_t(0));
        }

        ourPlayerReflectTable[i] = r;
    }
}

// Compute playfield mask table
void computePlayfieldMaskTable()
{
    int32_t x;

    // Compute playfield mask table for non-reflected mode
    for(x = 0; x < 160; ++x)
    {
        if(x < 16)
            ourPlayfieldTable[0][x] = 0x00001 << (x / 4);
        else if(x < 48)
            ourPlayfieldTable[0][x] = 0x00800 >> ((x - 16) / 4);
        else if(x < 80)
            ourPlayfieldTable[0][x] = 0x01000 << ((x - 48) / 4);
        else if(x < 96)
            ourPlayfieldTable[0][x] = 0x00001 << ((x - 80) / 4);
        else if(x < 128)
            ourPlayfieldTable[0][x] = 0x00800 >> ((x - 96) / 4);
        else if(x < 160)
            ourPlayfieldTable[0][x] = 0x01000 << ((x - 128) / 4);
    }

    // Compute playfield mask table for reflected mode
    for(x = 0; x < 160; ++x)
    {
        if(x < 16)
            ourPlayfieldTable[1][x] = 0x00001 << (x / 4);
        else if(x < 48)
            ourPlayfieldTable[1][x] = 0x00800 >> ((x - 16) / 4);
        else if(x < 80)
            ourPlayfieldTable[1][x] = 0x01000 << ((x - 48) / 4);
        else if(x < 112)
            ourPlayfieldTable[1][x] = 0x80000 >> ((x - 80) / 4);
        else if(x < 144)
            ourPlayfieldTable[1][x] = 0x00010 << ((x - 112) / 4);
        else if(x < 160)
            ourPlayfieldTable[1][x] = 0x00008 >> ((x - 144) / 4);
    }
}

void computePriorityEncoding()
{
    for(uint16_t x = 0; x < 2; ++x)
    {
        for(uint16_t enabled = 0; enabled < 256; ++enabled)
        {
            if(enabled & PriorityBit)
            {
                uint8_t color = 0;

                if((enabled & (P1Bit | M1Bit)) != 0)
                    color = 3;
                if((enabled & (P0Bit | M0Bit)) != 0)
                    color = 2;
                if((enabled & BLBit) != 0)
                    color = 1;
                if((enabled & PFBit) != 0)
                    color = 1;  // NOTE: Playfield has priority so ScoreBit isn't used

                ourPriorityEncoder[x][enabled] = color;
            }
            else
            {
                uint8_t color = 0;

                if((enabled & BLBit) != 0)
                    color = 1;
                if((enabled & PFBit) != 0)
                    color = (enabled & ScoreBit) ? ((x == 0) ? 2 : 3) : 1;
                if((enabled & (P1Bit | M1Bit)) != 0)
                    color = (color != 2) ? 3 : 2;
                if((enabled & (P0Bit | M0Bit)) != 0)
                    color = 2;

                ourPriorityEncoder[x][enabled] = color;
            }
        }
    }
}

CULE_ANNOTATION
uint32_t playfield_mask(const uint8_t side, const uint8_t x)
{
    uint32_t ret = (x < 16) * (0x00001 << (x / 4));

    ret |= ((x >= 16) && (x < 48)) * (0x00800 >> ((x - 16) / 4));
    ret |= ((x >= 48) && (x < 80)) * (0x01000 << ((x - 48) / 4));

    // Compute playfield mask table for non-reflected mode
    ret |= ((side == 0) && (x >= 80) && (x < 96)) * (0x00001 << ((x - 80) / 4));
    ret |= ((side == 0) && (x >= 96) && (x <128)) * (0x00800 >> ((x - 96) / 4));
    ret |= ((side == 0) && (x >=128) && (x <160)) * (0x01000 << ((x -128) / 4));

    // Compute playfield mask table for reflected mode
    ret |= ((side == 1) && (x >= 80) && (x <112)) * (0x80000 >> ((x - 80) / 4));
    ret |= ((side == 1) && (x >=112) && (x <144)) * (0x00010 << ((x -112) / 4));
    ret |= ((side == 1) && (x >=144) && (x <160)) * (0x00008 >> ((x -144) / 4));

    return ret;
}

CULE_ANNOTATION
bool missle_mask(const uint8_t align, const uint8_t number, const uint8_t size, int16_t x)
{
    x = (x + 320 - align) % 160;

    bool ret = (x >= 0) && (x < (1 << size));

    ret |= (number == 1) &&  (((x - 16) >= 0) && ((x - 16) < (1 << size)));
    ret |= (number == 2) &&  (((x - 32) >= 0) && ((x - 32) < (1 << size)));
    ret |= (number == 3) && ((((x - 16) >= 0) && ((x - 16) < (1 << size))) || ((((x - 32) >= 0) && ((x - 32) < (1 << size)))));
    ret |= (number == 4) &&  (((x - 64) >= 0) && ((x - 64) < (1 << size)));
    ret |= (number == 6) && ((((x - 32) >= 0) && ((x - 32) < (1 << size))) || ((((x - 64) >= 0) && ((x - 64) < (1 << size)))));

    return ret;
}

CULE_ANNOTATION
uint8_t player_mask(const uint8_t align, const bool enable, const uint8_t mode, int16_t x)
{
    x = (x + 320 - align) % 160;

    uint8_t shift = ((enable == 0) && (mode != 5) && (mode != 7) && (x >= 0) && (x < 8)) * (0x80 >> x);
    shift |= ((mode == 1 || mode == 3) && (((x - 16) >= 0) && ((x - 16) < 8))) * (0x80 >> (x - 16));
    shift |= ((mode == 2 || mode == 3 || mode == 6) && (((x - 32) >= 0) && ((x - 32) < 8))) * (0x80 >> (x - 32));
    shift |= ((mode == 5 && enable == 0) && ((x > 0) && (x <= 16))) * (0x80 >> ((x - 1) / 2));
    shift |= ((mode == 4 || mode == 6) && (((x - 64) >= 0) && ((x - 64) < 8))) * (0x80 >> (x - 64));
    shift |= ((mode == 7 && enable == 0) && ((x > 0) && (x <= 32))) * (0x80 >> ((x - 1) / 4));

    return shift;
}

CULE_ANNOTATION
bool ball_mask(const uint8_t align, const uint8_t size, int16_t x)
{
    x = (x + 320 - align) % 160;

    return (x >= 0) && (x < (1 << size));
}

CULE_ANNOTATION
uint8_t reflect_mask(uint8_t b)
{
    b = (b & 0xF0) >> 4 | (b & 0x0F) << 4;
    b = (b & 0xCC) >> 2 | (b & 0x33) << 2;
    b = (b & 0xAA) >> 1 | (b & 0x55) << 1;

    return b;
}

// Compute the collision decode table
uint16_t collision_mask(const uint8_t i)
{
    uint16_t mask = 0;

    mask |= ((i & M0Bit) && (i & P1Bit)) << 0;
    mask |= ((i & M0Bit) && (i & P0Bit)) << 1;
    mask |= ((i & M1Bit) && (i & P0Bit)) << 2;
    mask |= ((i & M1Bit) && (i & P1Bit)) << 3;
    mask |= ((i & P0Bit) && (i & PFBit)) << 4;
    mask |= ((i & P0Bit) && (i & BLBit)) << 5;
    mask |= ((i & P1Bit) && (i & PFBit)) << 6;
    mask |= ((i & P1Bit) && (i & BLBit)) << 7;
    mask |= ((i & M0Bit) && (i & PFBit)) << 8;
    mask |= ((i & M0Bit) && (i & BLBit)) << 9;
    mask |= ((i & M1Bit) && (i & PFBit)) << 10;
    mask |= ((i & M1Bit) && (i & BLBit)) << 11;
    mask |= ((i & BLBit) && (i & PFBit)) << 12;
    mask |= ((i & P0Bit) && (i & P1Bit)) << 13;
    mask |= ((i & M0Bit) && (i & M1Bit)) << 14;

    return mask;
}

int8_t position_mask(const uint8_t mode, const uint8_t oldx, const uint8_t newx)
{
    int8_t mask = ((newx >= oldx) && (newx < (oldx + 4))) * -1;
    mask |= ((mode == 0x01) && ((newx >= oldx + 16) && (newx < (oldx + 16 + 4)))) * -1;
    mask |= ((mode == 0x02) && ((newx >= oldx + 32) && (newx < (oldx + 32 + 4)))) * -1;

    mask |= (((mode <= 0x04) || (mode == 0x06)) && ((newx >= oldx + 4) && (newx < (oldx + 4 + 8)))) * 1;
    mask |= (((mode == 0x01) || (mode == 0x03)) && ((newx >= oldx + 16 + 4) && (newx < (oldx + 16 + 4 + 8)))) * 1;
    mask |= ((mode == 0x02) && ((newx >= oldx + 32 + 4) && (newx < (oldx + 32 + 4 + 8)))) * 1;
    mask |= ((mode == 0x05) && ((newx >= oldx + 4) && (newx < (oldx + 4 + 16)))) * 1;
    mask |= ((mode == 0x07) && ((newx >= oldx + 4) && (newx < (oldx + 4 + 32)))) * 1;

    return mask;
}

class tia_initialize
{
  public:
    tia_initialize()
    {
        // Compute all of the mask tables
        computeBallMaskTable();
        computeCollisionTable();
        computeMissleMaskTable();
        computePlayerMaskTable();
        computePlayerPositionResetWhenTable();
        computePlayerReflectTable();
        computePlayfieldMaskTable();
        computePriorityEncoding();
    }
};
static tia_initialize tia_startup;

} // end namespace atari
} // end namespace cule


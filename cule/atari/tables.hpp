#pragma once

#include <cule/config.hpp>

#include <cstdlib>

namespace cule
{
namespace atari
{

// Compute the ball mask table
void computeBallMaskTable();

// Compute the collision decode table
void computeCollisionTable();

// Compute the missle mask table
void computeMissleMaskTable();

// Compute the player mask table
void computePlayerMaskTable();

// Compute the player position reset when table
void computePlayerPositionResetWhenTable();

// Compute the player reflect table
void computePlayerReflectTable();

// Compute playfield mask table
void computePlayfieldMaskTable();

void computePriorityEncoding();

CULE_ANNOTATION
uint32_t playfield_mask(const uint8_t side, const uint8_t x);

CULE_ANNOTATION
bool missle_mask(const uint8_t align, const uint8_t number, const uint8_t size, int16_t x);

CULE_ANNOTATION
uint8_t player_mask(const uint8_t align, const bool enable, const uint8_t mode, int16_t x);

CULE_ANNOTATION
bool ball_mask(const uint8_t align, const uint8_t size, int16_t x);

CULE_ANNOTATION
uint8_t reflect_mask(uint8_t b);

// Compute the collision decode table
uint16_t collision_mask(const uint8_t i);

int8_t position_mask(const uint8_t mode, const uint8_t oldx, const uint8_t newx);

} // end namespace atari
} // end namespace cule

#include <cule/atari/tables.cpp>

#ifdef __CUDACC__
#include <cule/atari/cuda/tables.hpp>
#endif

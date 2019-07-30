#pragma once

#include <cule/config.hpp>
#include <cule/cuda/errchk.hpp>

#include <cule/atari/palettes.hpp>
#include <cule/atari/rom.hpp>
#include <cule/atari/tables.hpp>

#include <cstdlib>

#define COPY_DATA_FROM_VECTOR(ARR_NAME) CULE_ERRCHK(cudaMemcpyToSymbol(gpu_##ARR_NAME, cule::atari::ARR_NAME, sizeof(cule::atari::ARR_NAME)));

namespace cule
{
namespace atari
{
namespace cuda
{
__constant__ uint8_t gpu_rom[rom::MAX_ROM_SIZE];
__constant__ uint32_t gpu_NTSCPalette[256];
}

__device__ int8_t   gpu_ourPlayerPositionResetWhenTable[8][160][160];
__device__ uint8_t  gpu_ourBallMaskTable[4][4][320];
__device__ uint8_t  gpu_ourDisabledMaskTable[640];
__device__ uint8_t  gpu_ourPlayerMaskTable[4][2][8][320];
__device__ uint8_t  gpu_ourPlayerReflectTable[256];
__device__ uint8_t  gpu_ourMissleMaskTable[4][8][4][320];
__device__ uint8_t  gpu_ourPriorityEncoder[2][256];
__device__ uint16_t gpu_ourCollisionTable[64];
__device__ uint32_t gpu_ourPlayfieldTable[2][160];

__device__ bool     gpu_ourHMOVEBlankEnableCycles[76];
__device__ int16_t  gpu_ourPokeDelayTable[64];
__device__ int16_t  gpu_ourCompleteMotionTable[76][16];

void initialize_tables(const rom& cart)
{
    assert(cart.data() != nullptr);
    assert(cart.rom_size() != 0);

    CULE_ERRCHK(cudaMemcpyToSymbol(cule::atari::opcode::gpu_opdata, cule::atari::opcode::opdata, sizeof(cule::atari::opcode::M6502_OPCODE) * 256));
    CULE_ERRCHK(cudaMemcpyToSymbol(cuda::gpu_rom, cart.data(), sizeof(uint8_t) * cart.rom_size()));
    CULE_ERRCHK(cudaMemcpyToSymbol(cuda::gpu_NTSCPalette, cule::atari::NTSCPalette, sizeof(uint32_t) * 256));
    CULE_CUDA_PEEK_AND_SYNC;

    COPY_DATA_FROM_VECTOR(ourHMOVEBlankEnableCycles);
    COPY_DATA_FROM_VECTOR(ourPokeDelayTable);
    COPY_DATA_FROM_VECTOR(ourCompleteMotionTable);

    COPY_DATA_FROM_VECTOR(ourPlayerPositionResetWhenTable);
    COPY_DATA_FROM_VECTOR(ourBallMaskTable);
    COPY_DATA_FROM_VECTOR(ourDisabledMaskTable);
    COPY_DATA_FROM_VECTOR(ourPlayerMaskTable);
    COPY_DATA_FROM_VECTOR(ourPlayerReflectTable);
    COPY_DATA_FROM_VECTOR(ourMissleMaskTable);
    COPY_DATA_FROM_VECTOR(ourPriorityEncoder);
    COPY_DATA_FROM_VECTOR(ourCollisionTable);
    COPY_DATA_FROM_VECTOR(ourPlayfieldTable);
    CULE_CUDA_PEEK_AND_SYNC;
}
} // end namespace atari
} // end namespace cule

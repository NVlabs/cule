#pragma once

#include <cule/config.hpp>
#include <cule/atari/internals.hpp>

#include <array>
#include <sstream>
#include <fstream>
#include <zlib.h>

namespace cule
{
namespace atari
{

void writePNGChunk(std::ofstream& out, const char* type, uint8_t* data, int size)
{
    // Stuff the length/type into the buffer
    std::array<uint8_t,8> temp;
    temp[0] = size >> 24;
    temp[1] = size >> 16;
    temp[2] = size >> 8;
    temp[3] = size >> 0;
    temp[4] = type[0];
    temp[5] = type[1];
    temp[6] = type[2];
    temp[7] = type[3];

    // Write the header
    out.write(reinterpret_cast<const char*>(temp.data()), 8);

    // Append the actual data
    uint32_t crc = crc32(0, &temp[4], 4);
    if(size > 0)
    {
        out.write(reinterpret_cast<const char*>(data), size);
        crc = crc32(crc, data, size);
    }

    // Write the CRC
    temp[0] = crc >> 24;
    temp[1] = crc >> 16;
    temp[2] = crc >> 8;
    temp[3] = crc >> 0;

    out.write(reinterpret_cast<const char*>(temp.data()), 4);
}

void writePNGHeader(std::ofstream &out, const size_t dataWidth, const size_t height,  const size_t num_channels, bool doubleWidth)
{
    size_t width  = (doubleWidth ? 2 : 1) * dataWidth;

    // PNG file header
    std::array<uint8_t,8> header{ {137, 80, 78, 71, 13, 10, 26, 10} };
    out.write(reinterpret_cast<const char*>(header.data()), sizeof(header));

    // PNG IHDR
    std::array<uint8_t,13> ihdr;
    ihdr[0]  = (width >> 24) & 0xFF;   // width
    ihdr[1]  = (width >> 16) & 0xFF;
    ihdr[2]  = (width >>  8) & 0xFF;
    ihdr[3]  = (width >>  0) & 0xFF;
    ihdr[4]  = (height >> 24) & 0xFF;  // height
    ihdr[5]  = (height >> 16) & 0xFF;
    ihdr[6]  = (height >>  8) & 0xFF;
    ihdr[7]  = (height >>  0) & 0xFF;
    ihdr[8]  = 8;  // 8 bits per sample (24 bits per pixel)
    ihdr[9]  = num_channels == 1 ? 0 : 2;  // PNG_COLOR_TYPE_RGB
    ihdr[10] = 0;  // PNG_COMPRESSION_TYPE_DEFAULT
    ihdr[11] = 0;  // PNG_FILTER_TYPE_DEFAULT
    ihdr[12] = 0;  // PNG_INTERLACE_NONE
    writePNGChunk(out, "IHDR", ihdr.data(), sizeof(ihdr));
}

void writePNGData(std::ofstream &out, const size_t dataWidth, const size_t height, const uint8_t* data, const size_t num_channels, const bool doubleWidth)
{
    size_t scale = doubleWidth ? 2 : 1;
    size_t width = scale * dataWidth;

    // Fill the buffer with scanline data
    size_t rowbytes = num_channels * width;

    std::vector<uint8_t> buffer((rowbytes + 1) * height, 0);
    uint8_t* buf_ptr = buffer.data();

    for(size_t i = 0; i < height; i++)
    {
        *buf_ptr++ = 0; // first byte of row is filter type

        for(size_t j = 0; j < dataWidth; j++)
        {
            for(size_t k = 0; k < num_channels; k++)
            {
                int color = data[(i * num_channels * dataWidth) + (j * num_channels) + k];
                int jj = scale * j;
                buf_ptr[jj * num_channels + k] = color;

                // Double the pixel width, if so desired
                if (doubleWidth)
                {
                    jj = jj + 1;
                    buf_ptr[jj * num_channels + k] = color;
                }
            }
        }
        buf_ptr += rowbytes; // add pitch
    }

    size_t compmemsize = compressBound(buffer.size());
    std::vector<uint8_t> compmem(compmemsize, 0);

    // Compress the data with zlib
    assert(compress(compmem.data(), &compmemsize, buffer.data(), buffer.size()) == Z_OK);

    // Write the compressed framebuffer data
    writePNGChunk(out, "IDAT", compmem.data(), compmemsize);
}

void writePNGEnd(std::ofstream &out)
{
    // Finish up
    writePNGChunk(out, "IEND", 0, 0);
}

void generate_png(const uint8_t * buffer,
                  const std::string& filename,
                  const size_t num_channels,
                  const bool rescale,
                  const bool doubleWidth = true)
{
    // Open file for writing
    std::ofstream out(filename.c_str(), std::ios_base::binary);
    CULE_ASSERT(out.good(), "Could not open " << filename << " for writing\n");

    // Now write the PNG proper
    if(rescale)
    {
        writePNGHeader(out, 84, 84, num_channels, doubleWidth);
        writePNGData(out, 84, 84, buffer, num_channels, doubleWidth);
    }
    else
    {
        writePNGHeader(out, SCREEN_WIDTH, NTSC_SCREEN_HEIGHT, num_channels, doubleWidth);
        writePNGData(out, SCREEN_WIDTH, NTSC_SCREEN_HEIGHT, buffer, num_channels, doubleWidth);
    }

    writePNGEnd(out);

    out.close();
}

} // end namespace atari
} // end namespace cule


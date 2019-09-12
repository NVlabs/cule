#pragma once

#include <cule/config.hpp>

#include <cule/atari/frame_state.hpp>
#include <cule/atari/internals.hpp>
#include <cule/atari/tables.hpp>
#include <cule/atari/tia.hpp>

#include <string.h>

namespace cule
{
namespace atari
{
namespace preprocess
{

CULE_ANNOTATION
void reset(frame_state& s)
{
    UPDATE_FIELD(s.tiaFlags.asBitField(), FIELD_TIA_STATUS, 0);

    s.GRP = 0;
    s.HM = 0;
    s.PF = 0;
    s.POS = 0;
    s.Color = 0;

    s.lastHMOVEClock = 0;
    s.playfieldPriorityAndScore = 0;
    s.M0CosmicArkCounter = 0;

    s.CurrentGRP0 = 0;
    s.CurrentGRP1 = 0;

    s.CurrentPFMask = &playfield_accessor(0, 0);
    s.CurrentP0Mask = &player_mask_accessor(0, 0, 0, 0);
    s.CurrentP1Mask = &player_mask_accessor(0, 0, 0, 0);
    s.CurrentM0Mask = &missle_accessor(0, 0, 0, 0);
    s.CurrentM1Mask = &missle_accessor(0, 0, 0, 0);
    s.CurrentBLMask = &ball_accessor(0, 0, 0);
}

template<typename State_t>
CULE_ANNOTATION
int32_t currentScanline(State_t& s)
{
    return ((3 * s.cpuCycles) - s.clockWhenFrameStarted) / 228;
}

template<typename State_t>
CULE_ANNOTATION
bool is_ntsc(State_t& s)
{
    return s.tiaFlags[FLAG_TIA_IS_NTSC];
}

template<typename State_t>
CULE_ANNOTATION
uint8_t clocksThisLine(State_t& s)
{
    return ((3 * s.cpuCycles) - s.clockWhenFrameStarted) % 228;
}

template<typename State_t>
CULE_ANNOTATION
int32_t clockStartDisplay(State_t& s)
{
    return s.clockWhenFrameStarted + (228 * (30 + (4 * s.tiaFlags[FLAG_TIA_Y_SHIFT])));
}

template<typename State_t>
CULE_ANNOTATION
int32_t clockStopDisplay(State_t& s)
{
    return clockStartDisplay(s) + (228 *  (is_ntsc(s) ? 210 : 250));
}

CULE_ANNOTATION
void updateFrameScanline(frame_state& s,
                         const uint32_t& clocksToUpdate,
                         const int32_t& begin_pos,
                         const uint32_t& PF)
{
    // Calculate the ending frame pointer value
    uint8_t* ending = s.framePointer + clocksToUpdate;

    // See if we're in the vertical blank region
    if(s.tiaFlags[FLAG_TIA_VBLANK2])
    {
        memset(s.framePointer, 0, clocksToUpdate);
    }
    // Handle all other possible combinations
    else
    {
        int32_t end_pos = begin_pos + clocksToUpdate;

        for(int32_t hpos = begin_pos; hpos < end_pos; ++hpos)
        {
            uint8_t enabled = (PF & s.CurrentPFMask[hpos]) ? PFBit : 0;

            if(s.tiaFlags[FLAG_TIA_BLBit] && s.CurrentBLMask[hpos])
                enabled |= BLBit;

            if(s.CurrentGRP1 && (s.CurrentGRP1 & s.CurrentP1Mask[hpos]))
                enabled |= P1Bit;

            if(s.tiaFlags[FLAG_TIA_M1Bit] && s.CurrentM1Mask[hpos])
                enabled |= M1Bit;

            if(s.CurrentGRP0 && (s.CurrentGRP0 & s.CurrentP0Mask[hpos]))
                enabled |= P0Bit;

            if(s.tiaFlags[FLAG_TIA_M0Bit] && s.CurrentM0Mask[hpos])
                enabled |= M0Bit;

            int32_t shift = 8 * int(priority_accessor(hpos < 80 ? 0 : 1, enabled | s.playfieldPriorityAndScore));
            *s.framePointer++ = SELECT_FIELD(s.Color, 0xFF << shift);
        }
    }
    s.framePointer = ending;
}

CULE_ANNOTATION
void updateFrame(frame_state& s, const int32_t& clock)
{
    // See if we're in the nondisplayable portion of the screen or if
    // we've already updated this portion of the screen
    if((clock < clockStartDisplay(s)) ||
       (s.clockAtLastUpdate >= clockStopDisplay(s)) ||
       (s.clockAtLastUpdate >= clock))
    {
        return;
    }

    // Truncate the number of cycles to update to the stop display point
    int32_t temp_clock = min(clock, clockStopDisplay(s));

    const uint32_t PF = SELECT_FIELD(s.PF,  FIELD_PFALL);

    // Update frame one scanline at a time
    do
    {
        // Compute the number of clocks we're going to update
        int32_t clocksToUpdate = 0;

        // Remember how many clocks we are from the left side of the screen
        uint8_t clocksFromStartOfScanLine = (s.clockAtLastUpdate - s.clockWhenFrameStarted) % 228;
        uint8_t clocksToEndOfScanLine = 228 - clocksFromStartOfScanLine;

        // See if we're updating more than the current scanline
        if(temp_clock > (s.clockAtLastUpdate + clocksToEndOfScanLine))
        {
            // Yes, we have more than one scanline to update so finish current one
            clocksToUpdate = clocksToEndOfScanLine;
            s.clockAtLastUpdate += clocksToUpdate;
        }
        else
        {
            // No, so do as much of the current scanline as possible
            clocksToUpdate = temp_clock - s.clockAtLastUpdate;
            s.clockAtLastUpdate = temp_clock;
        }

        // Skip over as many horizontal blank clocks as we can
        if(clocksFromStartOfScanLine < HBLANK)
        {
            int32_t tmp = min(clocksToUpdate, HBLANK - clocksFromStartOfScanLine);

            clocksFromStartOfScanLine += tmp;
            clocksToUpdate -= tmp;
        }

        // Remember frame pointer in case HMOVE blanks need to be handled
        uint8_t* oldFramePointer = s.framePointer;

        // Update as much of the scanline as we can
        if(s.framePointer && (clocksToUpdate != 0))
        {
            updateFrameScanline(s, clocksToUpdate, clocksFromStartOfScanLine - HBLANK, PF);
        }

        // Handle HMOVE blanks if they are enabled
        if(s.tiaFlags[FLAG_TIA_HMOVE_ENABLE] && (clocksFromStartOfScanLine < (HBLANK + 8)))
        {
            if(s.framePointer)
            {
                const int32_t blanks = (HBLANK + 8) - clocksFromStartOfScanLine;
                memset(oldFramePointer, 0, blanks);
            }

            if((clocksToUpdate + clocksFromStartOfScanLine) >= (HBLANK + 8))
            {
                s.tiaFlags.clear(FLAG_TIA_HMOVE_ENABLE);
            }
        }

        if(clocksToEndOfScanLine == 228)
        {
            // Yes, so set PF mask based on current CTRLPF reflection state
            s.CurrentPFMask = &playfield_accessor(s.tiaFlags[FLAG_TIA_CTRLPF], 0);

            const uint8_t MODE0 = SELECT_FIELD(s.PF, FIELD_NUSIZ0_MODE);
            const uint8_t POSP0 = SELECT_FIELD(s.POS, FIELD_POSP0);
            s.CurrentP0Mask = &player_mask_accessor(POSP0 & 0x03, 0, MODE0, 160 - (POSP0 & 0xFC));

            const uint8_t MODE1 = SELECT_FIELD(s.PF, FIELD_NUSIZ1_MODE);
            const uint8_t POSP1 = SELECT_FIELD(s.POS, FIELD_POSP1);
            s.CurrentP1Mask = &player_mask_accessor(POSP1 & 0x03, 0, MODE1, 160 - (POSP1 & 0xFC));

            if(s.tiaFlags[FLAG_TIA_COSMIC_ARK])
            {
                static uint8_t m[4] = {18, 33, 0, 17};
                s.M0CosmicArkCounter = (s.M0CosmicArkCounter + 1) & 3;
                uint8_t POSM0 = SELECT_FIELD(s.POS, FIELD_POSM0);
                POSM0 = clamp(POSM0 - m[s.M0CosmicArkCounter]);
                UPDATE_FIELD(s.POS, FIELD_POSM0, POSM0);

                const uint8_t MODE = SELECT_FIELD(s.PF, FIELD_NUSIZ0_MODE);
                const uint8_t SIZE = SELECT_FIELD(s.PF, FIELD_NUSIZ0_SIZE);

                if(s.M0CosmicArkCounter == 1)
                {
                    s.CurrentM0Mask = &missle_accessor(POSM0 & 0x03, MODE, SIZE | 0x01, 160 - (POSM0 & 0xFC));
                }
                else if(s.M0CosmicArkCounter == 2)
                {
                    // Missle is disabled on this line
                    s.CurrentM0Mask = &disabled_accessor(0);
                }
                else
                {
                    s.CurrentM0Mask = &missle_accessor(POSM0 & 0x03, MODE, SIZE, 160 - (POSM0 & 0xFC));
                }
            }
        }
    }
    while(s.clockAtLastUpdate < temp_clock);
}

CULE_ANNOTATION
void poke(frame_state &s, const uint8_t& value, const uint8_t& addr)
{
    int16_t delay = poke_accessor(addr);

    // See if this is a poke to a PF register
    if(delay == -1)
    {
        const uint8_t x = (clocksThisLine(s) / 3) & 3;
        delay = x + (4 * (x < 2));
    }

    const bool read_only = addr == 0x80;
    delay *= !read_only;

    // Update frame to current CPU cycle before we make any changes!
    updateFrame(s, (3 * s.cpuCycles) + delay);

    if(read_only)
    {
        return;
    }

    switch(addr & 0x3F)
    {
        case ADR_VBLANK:    // Vertical blank set-clear
        {
            s.tiaFlags.change<FLAG_TIA_VBLANK1>((value & 0x80)==0x80);
            s.tiaFlags.change<FLAG_TIA_VBLANK2>((value & 0x02)==0x02);

            break;
        }
        case ADR_NUSIZ0:    // Number-size of player-missle 0
        {
            UPDATE_FIELD(s.PF, FIELD_NUSIZ0_MODE, value & 0x07);
            UPDATE_FIELD(s.PF, FIELD_NUSIZ0_SIZE, (value & 0x30) >> 4);

            const uint8_t MODE = value & 0x07;
            const uint8_t SIZE = (value & 0x30) >> 4;
            const uint8_t POSP0 = SELECT_FIELD(s.POS, FIELD_POSP0);
            const uint8_t POSM0 = SELECT_FIELD(s.POS, FIELD_POSM0);

            s.CurrentP0Mask = &player_mask_accessor(POSP0 & 0x03, 0, MODE, 160 - (POSP0 & 0xFC));
            s.CurrentM0Mask = &missle_accessor(POSM0 & 0x03, MODE, SIZE, 160 - (POSM0 & 0xFC));

            break;
        }
        case ADR_NUSIZ1:    // Number-size of player-missle 1
        {
            UPDATE_FIELD(s.PF, FIELD_NUSIZ1_MODE, value & 0x07);
            UPDATE_FIELD(s.PF, FIELD_NUSIZ1_SIZE, (value & 0x30) >> 4);

            const uint8_t MODE = value & 0x07;
            const uint8_t SIZE = (value & 0x30) >> 4;
            const uint8_t POSP1 = SELECT_FIELD(s.POS, FIELD_POSP1);
            const uint8_t POSM1 = SELECT_FIELD(s.POS, FIELD_POSM1);

            s.CurrentP1Mask = &player_mask_accessor(POSP1 & 0x03, 0, MODE, 160 - (POSP1 & 0xFC));
            s.CurrentM1Mask = &missle_accessor(POSM1 & 0x03, MODE, SIZE, 160 - (POSM1 & 0xFC));

            break;
        }
        case ADR_COLUP0:    // Color-Luminance Player 0
        {
            // if(s.ColorLossEnabled && (s.ScanlineCountForLastFrame & 0x01))
            // {
            //     color |= 0x01;
            // }
            UPDATE_FIELD(s.Color, FIELD_COLUP0, value & 0xFE);

            break;
        }
        case ADR_COLUP1:    // Color-Luminance Player 1
        {
            // if(s.ColorLossEnabled && (s.ScanlineCountForLastFrame & 0x01))
            // {
            //     color |= 0x01;
            // }
            UPDATE_FIELD(s.Color, FIELD_COLUP1, value & 0xFE);

            break;
        }
        case ADR_COLUPF:    // Color-Luminance Playfield
        {
            // if(s.ColorLossEnabled && (s.ScanlineCountForLastFrame & 0x01))
            // {
            //     color |= 0x01;
            // }
            UPDATE_FIELD(s.Color, FIELD_COLUPF, value & 0xFE);

            break;
        }
        case ADR_COLUBK:    // Color-Luminance Background
        {
            // if(s.ColorLossEnabled && (s.ScanlineCountForLastFrame & 0x01))
            // {
            //     color |= 0x01;
            // }
            UPDATE_FIELD(s.Color, FIELD_COLUBK, value & 0xFE);

            break;
        }
        case ADR_CTRLPF:    // Control Playfield, Ball size, Collisions
        {
            const uint8_t CTRLPF = (value & 0x30) >> 4;
            const uint8_t POSBL = SELECT_FIELD(s.HM,  FIELD_POSBL);

            s.tiaFlags.template change<FLAG_TIA_CTRLPF>((value & 0x01) == 0x01);
            s.CurrentPFMask = &playfield_accessor(s.tiaFlags[FLAG_TIA_CTRLPF], 0);
            s.CurrentBLMask = &ball_accessor(POSBL & 0x03, CTRLPF, 160 - (POSBL & 0xFC));

            // The playfield priority and score bits from the control register
            // are accessed when the frame is being drawn.  We precompute the
            // necessary value here so we can save time while drawing.
            s.playfieldPriorityAndScore = ((value & 0x06) << 5);

            break;
        }
        case ADR_REFP0:    // Reflect Player 0
        {
            // See if the reflection state of the player is being changed
            if(((value & 0x08)==0x08) ^ s.tiaFlags[FLAG_TIA_REFP0])
            {
                s.tiaFlags.template change<FLAG_TIA_REFP0>((value & 0x08)==0x08);
                s.CurrentGRP0 = reflect_mask(s.CurrentGRP0);
            }

            break;
        }
        case ADR_REFP1:    // Reflect Player 1
        {
            // See if the reflection state of the player is being changed
            if(((value & 0x08)==0x08) ^ s.tiaFlags[FLAG_TIA_REFP1])
            {
                s.tiaFlags.template change<FLAG_TIA_REFP1>((value & 0x08)==0x08);
                s.CurrentGRP1 = reflect_mask(s.CurrentGRP1);
            }

            break;
        }
        case ADR_PF0:    // Playfield register byte 0
        {
            UPDATE_FIELD(s.PF, FIELD_PF0, ((value >> 4) & 0x0F));
            uint32_t temp = SELECT_FIELD(s.PF, FIELD_PFALL);
            s.tiaFlags.change<FLAG_TIA_PFBit>(temp != 0);

            break;
        }
        case ADR_PF1:    // Playfield register byte 1
        {
            UPDATE_FIELD(s.PF, FIELD_PF1, value);
            uint32_t temp = SELECT_FIELD(s.PF, FIELD_PFALL);
            s.tiaFlags.change<FLAG_TIA_PFBit>(temp != 0);

            break;
        }
        case ADR_PF2:    // Playfield register byte 2
        {
            UPDATE_FIELD(s.PF, FIELD_PF2, value);
            uint32_t temp = SELECT_FIELD(s.PF, FIELD_PFALL);
            s.tiaFlags.change<FLAG_TIA_PFBit>(temp != 0);

            break;
        }
        case ADR_RESP0:    // Reset Player 0
        {
            const uint8_t hpos = clocksThisLine(s);
            const uint8_t newx = hpos < HBLANK ? 3 : (((hpos - HBLANK) + 5) % 160);

            const uint8_t MODE = SELECT_FIELD(s.PF, FIELD_NUSIZ0_MODE);
            const uint8_t POSP0 = SELECT_FIELD(s.POS, FIELD_POSP0);

            // Find out under what condition the player is being reset
            int8_t when = player_position_accessor(MODE, POSP0, newx);

            // Player is being reset during the display of one of its copies
            if(when == 1)
            {
                // So we go ahead and update the display before moving the player
                // TODO: The 11 should depend on how much of the player has already
                // been displayed.  Probably change table to return the amount to
                // delay by instead of just 1 (01/21/99).
                updateFrame(s, (3 * s.cpuCycles) + 11);
            }
            UPDATE_FIELD(s.POS, FIELD_POSP0, newx);

            s.CurrentP0Mask = &player_mask_accessor(newx & 0x03, (when == 0) || (when == 1), MODE, 160 - (newx & 0xFC));

            break;
        }
        case ADR_RESP1:    // Reset Player 1
        {
            uint8_t hpos = clocksThisLine(s);
            uint8_t newx = hpos < HBLANK ? 3 : (((hpos - HBLANK) + 5) % 160);

            const uint8_t MODE = SELECT_FIELD(s.PF, FIELD_NUSIZ1_MODE);
            const uint8_t POSP1 = SELECT_FIELD(s.POS, FIELD_POSP1);

            // Find out under what condition the player is being reset
            int8_t when = player_position_accessor(MODE, POSP1, newx);

            // Player is being reset during the display of one of its copies
            if(when == 1)
            {
                // So we go ahead and update the display before moving the player
                // TODO: The 11 should depend on how much of the player has already
                // been displayed.  Probably change table to return the amount to
                // delay by instead of just 1 (01/21/99).
                updateFrame(s, (3 * s.cpuCycles) + 11);
            }
            UPDATE_FIELD(s.POS, FIELD_POSP1, newx);

            s.CurrentP1Mask = &player_mask_accessor(newx & 0x03, (when == 0) || (when == 1), MODE, 160 - (newx & 0xFC));

            break;
        }
        case ADR_RESM0:    // Reset Missle 0
        {
            uint8_t hpos = clocksThisLine(s);
            uint8_t POSM0 = hpos < HBLANK ? 2 : (((hpos - HBLANK) + 4) % 160);
            UPDATE_FIELD(s.POS, FIELD_POSM0, POSM0);

            uint32_t clock = 3 * s.cpuCycles;

            // TODO: Remove the following special hack for Pitfall II by
            // figuring out what really happens when Reset Missle
            // occurs 3 cycles after an HMOVE (04/13/02).
            if(((clock - s.lastHMOVEClock) == (20 * 3)) && (hpos == 69))
            {
                POSM0 = 8;
                UPDATE_FIELD(s.POS, FIELD_POSM0, 8);
            }

            const uint8_t MODE = SELECT_FIELD(s.PF, FIELD_NUSIZ0_MODE);
            const uint8_t SIZE = SELECT_FIELD(s.PF, FIELD_NUSIZ0_SIZE);
            s.CurrentM0Mask = &missle_accessor(POSM0 & 0x03, MODE, SIZE, 160 - (POSM0 & 0xFC));

            break;
        }
        case ADR_RESM1:    // Reset Missle 1
        {
            uint8_t hpos = clocksThisLine(s);
            uint8_t POSM1 = hpos < HBLANK ? 2 : (((hpos - HBLANK) + 4) % 160);
            UPDATE_FIELD(s.POS, FIELD_POSM1, POSM1);

            uint32_t clock = 3 * s.cpuCycles;

            // TODO: Remove the following special hack for Pitfall II by
            // figuring out what really happens when Reset Missle
            // occurs 3 cycles after an HMOVE (04/13/02).
            if(((clock - s.lastHMOVEClock) == (3 * 3)) && (hpos == 18))
            {
                POSM1 = 3;
                UPDATE_FIELD(s.POS, FIELD_POSM1, 3);
            }

            const uint8_t MODE = SELECT_FIELD(s.PF, FIELD_NUSIZ1_MODE);
            const uint8_t SIZE = SELECT_FIELD(s.PF, FIELD_NUSIZ1_SIZE);
            s.CurrentM1Mask = &missle_accessor(POSM1 & 0x03, MODE, SIZE, 160 - (POSM1 & 0xFC));

            break;
        }
        case ADR_RESBL:    // Reset Ball
        {
            uint8_t hpos = clocksThisLine(s);
            uint8_t POSBL = hpos < HBLANK ? 2 : (((hpos - HBLANK) + 4) % 160);

            uint32_t clock = 3 * s.cpuCycles;

            // TODO: Remove the following special hack for Escape from the
            // Mindmaster by figuring out what really happens when Reset Ball
            // occurs 18 cycles after an HMOVE (01/09/99).
            if(((clock - s.lastHMOVEClock) == (18 * 3)) &&
                    ((hpos == 60) || (hpos == 69)))
            {
                POSBL = 10;
            }
            // TODO: Remove the following special hack for Decathlon by
            // figuring out what really happens when Reset Ball
            // occurs 3 cycles after an HMOVE (04/13/02).
            else if(((clock - s.lastHMOVEClock) == (3 * 3)) && (hpos == 18))
            {
                POSBL = 3;
            }
            // TODO: Remove the following special hack for Robot Tank by
            // figuring out what really happens when Reset Ball
            // occurs 7 cycles after an HMOVE (04/13/02).
            else if(((clock - s.lastHMOVEClock) == (7 * 3)) && (hpos == 30))
            {
                POSBL = 6;
            }
            // TODO: Remove the following special hack for Hole Hunter by
            // figuring out what really happens when Reset Ball
            // occurs 6 cycles after an HMOVE (04/13/02).
            else if(((clock - s.lastHMOVEClock) == (6 * 3)) && (hpos == 27))
            {
                POSBL = 5;
            }
            UPDATE_FIELD(s.HM, FIELD_POSBL, POSBL);

            const uint8_t CTRLPF = SELECT_FIELD(s.PF,  FIELD_CTRLPF);
            s.CurrentBLMask = &ball_accessor(POSBL & 0x03, CTRLPF, 160 - (POSBL & 0xFC));

            break;
        }
        case ADR_GRP0:    // Graphics Player 0
        {
            // Set player 0 graphics
            UPDATE_FIELD(s.GRP, FIELD_GRP0, value);

            // Copy player 1 graphics into its delayed register
            uint8_t temp = SELECT_FIELD(s.GRP, FIELD_GRP1);
            UPDATE_FIELD(s.GRP, FIELD_DGRP1, temp);

            // Get the "current" data for GRP0 base on delay register and reflect
            uint8_t grp0 = s.tiaFlags[FLAG_TIA_VDELP0] ? SELECT_FIELD(s.GRP, FIELD_DGRP0) : SELECT_FIELD(s.GRP, FIELD_GRP0);
            s.CurrentGRP0 = s.tiaFlags[FLAG_TIA_REFP0] ? reflect_mask(grp0) : grp0;

            // Get the "current" data for GRP1 base on delay register and reflect
            uint8_t grp1 = s.tiaFlags[FLAG_TIA_VDELP1] ? SELECT_FIELD(s.GRP, FIELD_DGRP1) : SELECT_FIELD(s.GRP, FIELD_GRP1);
            s.CurrentGRP1 = s.tiaFlags[FLAG_TIA_REFP1] ? reflect_mask(grp1) : grp1;

            // Set enabled object bits
            s.tiaFlags.template change<FLAG_TIA_P0Bit>(grp0 != 0);
            s.tiaFlags.template change<FLAG_TIA_P1Bit>(grp1 != 0);

            break;
        }
        case ADR_GRP1:    // Graphics Player 1
        {
            // Set player 1 graphics
            UPDATE_FIELD(s.GRP, FIELD_GRP1, value);

            // Copy player 0 graphics into its delayed register
            uint8_t temp = SELECT_FIELD(s.GRP, FIELD_GRP0);
            UPDATE_FIELD(s.GRP, FIELD_DGRP0, temp);

            // Copy ball graphics into its delayed register
            s.tiaFlags.template change<FLAG_TIA_DENABL>(s.tiaFlags[FLAG_TIA_ENABL]);

            // Get the "current" data for GRP0 base on delay register
            uint8_t grp0 = s.tiaFlags[FLAG_TIA_VDELP0] ? SELECT_FIELD(s.GRP, FIELD_DGRP0) : SELECT_FIELD(s.GRP, FIELD_GRP0);
            s.CurrentGRP0 = s.tiaFlags[FLAG_TIA_REFP0] ? reflect_mask(grp0) : grp0;

            // Get the "current" data for GRP1 base on delay register
            uint8_t grp1 = s.tiaFlags[FLAG_TIA_VDELP1] ? SELECT_FIELD(s.GRP, FIELD_DGRP1) : SELECT_FIELD(s.GRP, FIELD_GRP1);
            s.CurrentGRP1 = s.tiaFlags[FLAG_TIA_REFP1] ? reflect_mask(grp1) : grp1;

            // Set enabled object bits
            s.tiaFlags.template change<FLAG_TIA_P0Bit>(grp0 != 0);
            s.tiaFlags.template change<FLAG_TIA_P1Bit>(grp1 != 0);
            s.tiaFlags.template change<FLAG_TIA_BLBit>(s.tiaFlags[FLAG_TIA_VDELBL] ? s.tiaFlags[FLAG_TIA_DENABL] : s.tiaFlags[FLAG_TIA_ENABL]);

            break;
        }
        case ADR_ENAM0:    // Enable Missile 0 graphics
        {
            s.tiaFlags.change<FLAG_TIA_ENAM0>((value & 0x02)==0x02);
            s.tiaFlags.change<FLAG_TIA_M0Bit>(s.tiaFlags[FLAG_TIA_ENAM0] && !s.tiaFlags[FLAG_TIA_RESMP0]);

            break;
        }
        case ADR_ENAM1:    // Enable Missile 1 graphics
        {
            s.tiaFlags.change<FLAG_TIA_ENAM1>((value & 0x02)==0x02);
            s.tiaFlags.change<FLAG_TIA_M1Bit>(s.tiaFlags[FLAG_TIA_ENAM1] && !s.tiaFlags[FLAG_TIA_RESMP1]);

            break;
        }
        case ADR_ENABL:    // Enable Ball graphics
        {
            s.tiaFlags.change<FLAG_TIA_ENABL>((value & 0x02)==0x02);
            s.tiaFlags.change<FLAG_TIA_BLBit>(s.tiaFlags[FLAG_TIA_VDELBL] ? s.tiaFlags[FLAG_TIA_DENABL] : s.tiaFlags[FLAG_TIA_ENABL]);

            break;
        }
        case ADR_HMP0:    // Horizontal Motion Player 0
        {
            uint8_t temp = value >> 4;
            UPDATE_FIELD(s.HM, FIELD_HMP0, temp);

            break;
        }
        case ADR_HMP1:    // Horizontal Motion Player 1
        {
            uint8_t temp = value >> 4;
            UPDATE_FIELD(s.HM, FIELD_HMP1, temp);

            break;
        }
        case ADR_HMM0:    // Horizontal Motion Missle 0
        {
            uint8_t temp = value >> 4;

            // Should we enabled TIA M0 "bug" used for stars in Cosmic Ark?
            if(((3 * s.cpuCycles) == (s.lastHMOVEClock + 21 * 3)) && (SELECT_FIELD(s.HM, FIELD_HMM0) == 7) && (temp == 6))
            {
                s.tiaFlags.set(FLAG_TIA_COSMIC_ARK);
                s.M0CosmicArkCounter = 0;
            }

            UPDATE_FIELD(s.HM, FIELD_HMM0, temp);

            break;
        }
        case ADR_HMM1:    // Horizontal Motion Missle 1
        {
            uint8_t temp = value >> 4;
            UPDATE_FIELD(s.HM, FIELD_HMM1, temp);

            break;
        }
        case ADR_HMBL:    // Horizontal Motion Ball
        {
            uint8_t temp = value >> 4;
            UPDATE_FIELD(s.HM, FIELD_HMBL, temp);

            break;
        }
        case ADR_VDELP0:    // Vertial Delay Player 0
        {
            s.tiaFlags.template change<FLAG_TIA_VDELP0>((value & 0x01)==0x01);

            uint8_t grp0 = s.tiaFlags[FLAG_TIA_VDELP0] ? SELECT_FIELD(s.GRP, FIELD_DGRP0) : SELECT_FIELD(s.GRP, FIELD_GRP0);
            s.CurrentGRP0 = s.tiaFlags[FLAG_TIA_REFP0] ? reflect_mask(grp0) : grp0;
            s.tiaFlags.template change<FLAG_TIA_P0Bit>(grp0 != 0);

            break;
        }
        case ADR_VDELP1:    // Vertial Delay Player 1
        {
            s.tiaFlags.template change<FLAG_TIA_VDELP1>((value & 0x01)==0x01);

            uint8_t grp1 = s.tiaFlags[FLAG_TIA_VDELP1] ? SELECT_FIELD(s.GRP, FIELD_DGRP1) : SELECT_FIELD(s.GRP, FIELD_GRP1);
            s.CurrentGRP1 = s.tiaFlags[FLAG_TIA_REFP1] ? reflect_mask(grp1) : grp1;
            s.tiaFlags.template change<FLAG_TIA_P1Bit>(grp1 != 0);

            break;
        }
        case ADR_VDELBL:    // Vertial Delay Ball
        {
            s.tiaFlags.change<FLAG_TIA_VDELBL>((value & 0x01)==0x01);
            s.tiaFlags.change<FLAG_TIA_BLBit>(s.tiaFlags[FLAG_TIA_VDELBL] ? s.tiaFlags[FLAG_TIA_DENABL] : s.tiaFlags[FLAG_TIA_ENABL]);

            break;
        }
        case ADR_RESMP0:    // Reset missle 0 to player 0
        {
            if(s.tiaFlags[FLAG_TIA_RESMP0] && !(value & 0x02))
            {
                const uint8_t MODE = SELECT_FIELD(s.PF, FIELD_NUSIZ0_MODE);
                const uint8_t SIZE = SELECT_FIELD(s.PF, FIELD_NUSIZ0_SIZE);

                uint16_t middle;

                if(MODE==0x05)
                    middle = 8;
                else if(MODE==0x07)
                    middle = 16;
                else
                    middle = 4;

                const uint8_t POSM0 = (SELECT_FIELD(s.POS, FIELD_POSP0) + middle) % 160;
                UPDATE_FIELD(s.POS, FIELD_POSM0, POSM0);
                s.CurrentM0Mask = &missle_accessor(POSM0 & 0x03, MODE, SIZE, 160 - (POSM0 & 0xFC));
            }

            s.tiaFlags.template change<FLAG_TIA_RESMP0>((value & 0x02)==0x02);
            s.tiaFlags.template change<FLAG_TIA_M0Bit>(s.tiaFlags[FLAG_TIA_ENAM0] && !s.tiaFlags[FLAG_TIA_RESMP0]);

            break;
        }
        case ADR_RESMP1:    // Reset missle 1 to player 1
        {
            if(s.tiaFlags[FLAG_TIA_RESMP1] && !(value & 0x02))
            {
                const uint8_t MODE = SELECT_FIELD(s.PF, FIELD_NUSIZ1_MODE);
                const uint8_t SIZE = SELECT_FIELD(s.PF, FIELD_NUSIZ1_SIZE);

                uint16_t middle;

                if(MODE==0x05)
                    middle = 8;
                else if(MODE==0x07)
                    middle = 16;
                else
                    middle = 4;

                const uint8_t POSM1 = (SELECT_FIELD(s.POS, FIELD_POSP1) + middle) % 160;
                UPDATE_FIELD(s.POS, FIELD_POSM1, POSM1);
                s.CurrentM1Mask = &missle_accessor(POSM1 & 0x03, MODE, SIZE, 160 - (POSM1 & 0xFC));
            }

            s.tiaFlags.template change<FLAG_TIA_RESMP1>((value & 0x02)==0x02);
            s.tiaFlags.template change<FLAG_TIA_M1Bit>(s.tiaFlags[FLAG_TIA_ENAM1] && !s.tiaFlags[FLAG_TIA_RESMP1]);

            break;
        }
        case ADR_HMOVE:    // Apply horizontal motion
        {
            // Figure out what cycle we're at
            uint8_t x = clocksThisLine(s) / 3;

            // See if we need to enable the HMOVE blank bug
            s.tiaFlags.template change<FLAG_TIA_HMOVE_ENABLE>(s.tiaFlags[FLAG_TIA_HMOVE_ALLOW] && hmove_accessor(x));

            const uint8_t NUSIZ0_MODE = SELECT_FIELD(s.PF, FIELD_NUSIZ0_MODE);
            const uint8_t NUSIZ1_MODE = SELECT_FIELD(s.PF, FIELD_NUSIZ1_MODE);
            const uint8_t NUSIZ0_SIZE = SELECT_FIELD(s.PF, FIELD_NUSIZ0_SIZE);
            const uint8_t NUSIZ1_SIZE = SELECT_FIELD(s.PF, FIELD_NUSIZ1_SIZE);
            const uint8_t CTRLPF = SELECT_FIELD(s.PF,  FIELD_CTRLPF);

            uint8_t POSP0 = clamp(SELECT_FIELD(s.POS, FIELD_POSP0) + motion_accessor(x, SELECT_FIELD(s.HM, FIELD_HMP0)));
            UPDATE_FIELD(s.POS, FIELD_POSP0, POSP0);
            s.CurrentP0Mask = &player_mask_accessor(POSP0 & 0x03, 0, NUSIZ0_MODE, 160 - (POSP0 & 0xFC));

            uint8_t POSP1 = clamp(SELECT_FIELD(s.POS, FIELD_POSP1) + motion_accessor(x, SELECT_FIELD(s.HM, FIELD_HMP1)));
            UPDATE_FIELD(s.POS, FIELD_POSP1, POSP1);
            s.CurrentP1Mask = &player_mask_accessor(POSP1 & 0x03, 0, NUSIZ1_MODE, 160 - (POSP1 & 0xFC));

            uint8_t POSM0 = clamp(SELECT_FIELD(s.POS, FIELD_POSM0) + motion_accessor(x, SELECT_FIELD(s.HM, FIELD_HMM0)));
            UPDATE_FIELD(s.POS, FIELD_POSM0, POSM0);
            s.CurrentM0Mask = &missle_accessor(POSM0 & 0x03, NUSIZ0_MODE, NUSIZ0_SIZE, 160 - (POSM0 & 0xFC));

            uint8_t POSM1 = clamp(SELECT_FIELD(s.POS, FIELD_POSM1) + motion_accessor(x, SELECT_FIELD(s.HM, FIELD_HMM1)));
            UPDATE_FIELD(s.POS, FIELD_POSM1, POSM1);
            s.CurrentM1Mask = &missle_accessor(POSM1 & 0x03, NUSIZ1_MODE, NUSIZ1_SIZE, 160 - (POSM1 & 0xFC));

            uint8_t POSBL = clamp(SELECT_FIELD(s.HM,  FIELD_POSBL) + motion_accessor(x, SELECT_FIELD(s.HM, FIELD_HMBL)));
            UPDATE_FIELD(s.HM,  FIELD_POSBL, POSBL);
            s.CurrentBLMask = &ball_accessor(POSBL & 0x03, CTRLPF, 160 - (POSBL & 0xFC));

            s.lastHMOVEClock = 3 * s.cpuCycles;

            // Disable TIA M0 "bug" used for stars in Cosmic ark
            s.tiaFlags.clear(FLAG_TIA_COSMIC_ARK);

            break;
        }
        case ADR_HMCLR:    // Clear horizontal motion registers
        {
            UPDATE_FIELD(s.HM, FIELD_HMALL, 0);

            break;
        }
        default:
        {
            break;
        }
    }
}

CULE_ANNOTATION
bool state_to_buffer(frame_state& s)
{
    if((*s.srcBuffer++ & 0xFF) != 0xFE)
    {
        return false;
    }

    s.clockWhenFrameStarted = -(*s.srcBuffer++);
    s.clockAtLastUpdate = clockStartDisplay(s);

    uint32_t entry = *(s.srcBuffer++);

    while((entry & 0xFF) != 0xFD)
    {
        s.cpuCycles = entry >> 16;
        uint8_t value = (entry >> 8) & 0xFF;
        uint8_t addr  = entry & 0x3F;

        // printf("cycles: %d, value: 0x%02X, address: 0x%02X\n", s.cpuCycles, value, addr);

        poke(s, value, addr);
        entry = *(s.srcBuffer++);
    }

    return true;
}

} // end namespace preprocess
} // end namespace atari
} // end namespace cule


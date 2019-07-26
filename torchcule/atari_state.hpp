#pragma once

#include <array>
#include <cstdint>
#include <string>

struct AtariState
{
    AtariState(){}

    // ale vars
    int32_t left_paddle;
    int32_t right_paddle;
    int32_t frame_number;
    int32_t episode_frame_number;
    int32_t string_length;
    bool save_system;
    std::string md5;

    // system vars
    int32_t cycles;

    // m6502 vars
    int32_t	A; // accumulator
    int32_t	X, Y; // index

    int32_t SP; // stack pointer
    int32_t PC; // program counter
    int32_t IR; // interrupt

    bool N;
    bool V;
    bool B;
    bool D;
    bool I;
    bool notZ;
    bool C;

    int32_t executionStatus;

    // m6532 vars
    std::array<int32_t, 256> ram;

    int32_t timer;
    int32_t intervalShift;
    int32_t cyclesWhenTimerSet;
    int32_t cyclesWhenInterruptReset;
    bool timerReadAfterInterrupt;
    int32_t DDRA;
    int32_t DDRB;

    // TIA vars
    int32_t clockWhenFrameStarted;
    int32_t clockStartDisplay;
    int32_t clockStopDisplay;
    int32_t clockAtLastUpdate;
    int32_t clocksToEndOfScanLine;
    int32_t scanlineCountForLastFrame;
    int32_t currentScanline;
    int32_t VSYNCFinishClock;
    int32_t enabledObjects;
    int32_t VSYNC;
    int32_t VBLANK;
    int32_t NUSIZ0;
    int32_t NUSIZ1;
    int32_t COLUP0;
    int32_t COLUP1;
    int32_t COLUPF;
    int32_t COLUBK;
    int32_t CTRLPF;
    int32_t playfieldPriorityAndScore;
    bool REFP0;
    bool REFP1;
    int32_t PF;
    int32_t GRP0;
    int32_t GRP1;
    int32_t DGRP0;
    int32_t DGRP1;
    int32_t ENAM0;
    int32_t ENAM1;
    int32_t ENABL;
    int32_t DENABL;
    int32_t HMP0;
    int32_t HMP1;
    int32_t HMM0;
    int32_t HMM1;
    int32_t HMBL;
    int32_t VDELP0;
    int32_t VDELP1;
    int32_t VDELBL;
    int32_t RESMP0;
    int32_t RESMP1;
    int32_t collision;
    int32_t POSP0;
    int32_t POSP1;
    int32_t POSM0;
    int32_t POSM1;
    int32_t POSBL;
    int32_t currentGRP0;
    int32_t currentGRP1;
    int32_t lastHMOVEClock;
    int32_t HMOVEBlankEnabled;
    int32_t M0CosmicArkMotionEnabled;
    int32_t M0CosmicArkCounter;
    int32_t dumpEnabled;
    int32_t dumpDisabledCycle;

    int32_t bank;
    int32_t reward;
    int32_t score;
    bool terminal;
    bool started;
    int32_t lives;
    int32_t points;
    int32_t last_lives;
};

struct encode_states_functor;
struct decode_states_functor;


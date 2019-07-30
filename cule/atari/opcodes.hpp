#pragma once

namespace cule
{
namespace atari
{
namespace opcode
{

/* 6502 addressing modes */
enum M6502_ADDRMODE : uint8_t
{
    ADR_ABS=0,
    ADR_ABSX,
    ADR_ABSY,
    ADR_IMM,
    ADR_IMP,
    ADR_ACC=ADR_IMP,
    ADR_INDABSX,
    ADR_IND, // JMP bug
    ADR_INDX,
    ADR_PREIDXIND=ADR_INDX,
    ADR_INDY,
    ADR_POSTIDXIND=ADR_INDY,
    ADR_INDZP,
    ADR_REL,
    ADR_ZP,
    ADR_ZPX,
    ADR_ZPY,
    _ADR_MAX,
    _ADR_INVALID=255
};

/* 6502 instruction set */
enum M6502_INST : uint8_t
{
    INS_ADC=0,
    INS_AND,
    INS_ASL,
    INS_ASLA,
    INS_BCC,
    INS_BCS,
    INS_BEQ,
    INS_BIT,
    INS_BMI,
    INS_BNE,
    INS_BPL,
    INS_BRK,
    INS_BVC,
    INS_BVS,
    INS_CLC,
    INS_CLD,
    INS_CLI,
    INS_CLV,
    INS_CMP,
    INS_CPX,
    INS_CPY,
    INS_DEC,
    INS_DEA,
    INS_DEX,
    INS_DEY,
    INS_EOR,
    INS_INC,
    INS_INX,
    INS_INY,
    INS_JMP,
    INS_JSR,
    INS_LDA,
    INS_LDX,
    INS_LDY,
    INS_LSR,
    INS_LSRA,
    INS_NOP,
    INS_ORA,
    INS_PHA,
    INS_PHP,
    INS_PLA,
    INS_PLP,
    INS_ROL,
    INS_ROLA,
    INS_ROR,
    INS_RORA,
    INS_RTI,
    INS_RTS,
    INS_SBC,
    INS_SEC,
    INS_SED,
    INS_SEI,
    INS_STA,
    INS_STX,
    INS_STY,
    INS_TAX,
    INS_TAY,
    INS_TSX,
    INS_TXA,
    INS_TXS,
    INS_TYA,

    // illegal opcodes
    INS_ANC,
    INS_ANE,
    INS_ASR,
    INS_ARR,
    INS_DCP,
    INS_ISB,
    INS_ISC,
    INS_LAS,
    INS_LAX,
    INS_LXA,
    INS_RLA,
    INS_RRA,
    INS_SAX,
    INS_SBX,
    INS_SHA,
    INS_SHS,
    INS_SHX,
    INS_SHY,
    INS_SLO,
    INS_SRE,
    // filler
    INS_NA,

    _INS_MAX,
    _INS_INVALID=255
};

/* 6502 opcode entries, each occupies exactly 4 bytes */
struct M6502_OPCODE
{
    M6502_INST inst;
    uint8_t cycles;
    M6502_ADDRMODE addrmode;
    uint8_t size;
};

extern M6502_OPCODE opdata[256];
extern bool usualOp[256];
extern const char* adrmodeDesc[(int)_ADR_MAX];
extern const char* instructionName[(int)_INS_MAX];

CULE_ANNOTATION
M6502_OPCODE decode(const opcode_t& opcode);

CULE_ANNOTATION

bool usual(const bool* usualOp, const opcode_t& opcode);

const char* instName(const M6502_INST inst);

const char* instName(const opcode_t opcode);

const char* explainAddrMode(const M6502_ADDRMODE adrmode);

} // end namespace opcode
} // end namespace atari
} // end namespace cule

#include <cule/atari/opcodes.cpp>


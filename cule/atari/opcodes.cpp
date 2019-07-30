// local header files
#include <cule/macros.hpp>

#include <cule/atari/internals.hpp>
#include <cule/atari/types/types.hpp>

#include <cassert>
#include <cstring>

namespace cule
{
namespace atari
{
namespace opcode
{

M6502_OPCODE opdata[256];
bool usualOp[256]= {false};
const char* adrmodeDesc[(int)_ADR_MAX];
const char* instructionName[(int)_INS_MAX];

#ifdef __CUDACC__
__constant__ M6502_OPCODE gpu_opdata[256];
#endif

CULE_ANNOTATION
M6502_OPCODE* accessor()
{
    #ifdef __CUDA_ARCH__
    return gpu_opdata;
    #else
    return opdata;
    #endif
}

CULE_ANNOTATION
M6502_OPCODE decode(const opcode_t& opcode)
{
    return accessor()[opcode];
}

CULE_ANNOTATION
bool usual(const bool* OpTable, const opcode_t& opcode)
{
    return OpTable[opcode];
}

const char* instName(const M6502_INST inst)
{
    return instructionName[(int)inst];
}

const char* instName(const opcode_t opcode)
{
    return instName(accessor()[opcode].inst);
}

const char* explainAddrMode(const M6502_ADDRMODE adrmode)
{
    return adrmodeDesc[(int)adrmode];
}

static void initStrings()
{
    adrmodeDesc[(int)ADR_ABS]="Absolute";
    adrmodeDesc[(int)ADR_ABSX]="Absolute,X";
    adrmodeDesc[(int)ADR_ABSY]="Absolute,Y";
    adrmodeDesc[(int)ADR_IMM]="Immediate";
    adrmodeDesc[(int)ADR_IMP]="Implied";
    adrmodeDesc[(int)ADR_INDABSX]="ADR_INDABSX JMP 7C";
    adrmodeDesc[(int)ADR_IND]="Indirect Absolute (JMP)";
    adrmodeDesc[(int)ADR_INDX]="(IND,X) Preindexed Indirect";
    adrmodeDesc[(int)ADR_INDY]="(IND),Y Post-indexed Indirect mode";
    adrmodeDesc[(int)ADR_INDZP]="ADR_INDZP";
    adrmodeDesc[(int)ADR_REL]="Relative (Branch)";
    adrmodeDesc[(int)ADR_ZP]="Zero Page";
    adrmodeDesc[(int)ADR_ZPX]="Zero Page,X";
    adrmodeDesc[(int)ADR_ZPY]="Zero Page,Y";

    instructionName[(int)INS_ADC]="ADC";
    instructionName[(int)INS_AND]="AND";
    instructionName[(int)INS_ASL]="ASL";
    instructionName[(int)INS_ASLA]="ASLA";
    instructionName[(int)INS_BCC]="BCC";
    instructionName[(int)INS_BCS]="BCS";
    instructionName[(int)INS_BEQ]="BEQ";
    instructionName[(int)INS_BIT]="BIT";
    instructionName[(int)INS_BMI]="BMI";
    instructionName[(int)INS_BNE]="BNE";
    instructionName[(int)INS_BPL]="BPL";
    instructionName[(int)INS_BRK]="BRK";
    instructionName[(int)INS_BVC]="BVC";
    instructionName[(int)INS_BVS]="BVS";
    instructionName[(int)INS_CLC]="CLC";
    instructionName[(int)INS_CLD]="CLD";
    instructionName[(int)INS_CLI]="CLI";
    instructionName[(int)INS_CLV]="CLV";
    instructionName[(int)INS_CMP]="CMP";
    instructionName[(int)INS_CPX]="CPX";
    instructionName[(int)INS_CPY]="CPY";
    instructionName[(int)INS_DEC]="DEC";
    instructionName[(int)INS_DEA]="DEA";
    instructionName[(int)INS_DEX]="DEX";
    instructionName[(int)INS_DEY]="DEY";
    instructionName[(int)INS_EOR]="EOR";
    instructionName[(int)INS_INC]="INC";
    instructionName[(int)INS_INX]="INX";
    instructionName[(int)INS_INY]="INY";
    instructionName[(int)INS_JMP]="JMP";
    instructionName[(int)INS_JSR]="JSR";
    instructionName[(int)INS_LDA]="LDA";
    instructionName[(int)INS_LDX]="LDX";
    instructionName[(int)INS_LDY]="LDY";
    instructionName[(int)INS_LSR]="LSR";
    instructionName[(int)INS_LSRA]="LSRA";
    instructionName[(int)INS_NOP]="NOP";
    instructionName[(int)INS_ORA]="ORA";
    instructionName[(int)INS_PHA]="PHA";
    instructionName[(int)INS_PHP]="PHP";
    instructionName[(int)INS_PLA]="PLA";
    instructionName[(int)INS_PLP]="PLP";
    instructionName[(int)INS_ROL]="ROL";
    instructionName[(int)INS_ROLA]="ROLA";
    instructionName[(int)INS_ROR]="ROR";
    instructionName[(int)INS_RORA]="RORA";
    instructionName[(int)INS_RTI]="RTI";
    instructionName[(int)INS_RTS]="RTS";
    instructionName[(int)INS_SBC]="SBC";
    instructionName[(int)INS_SEC]="SEC";
    instructionName[(int)INS_SED]="SED";
    instructionName[(int)INS_SEI]="SEI";
    instructionName[(int)INS_STA]="STA";
    instructionName[(int)INS_STX]="STX";
    instructionName[(int)INS_STY]="STY";
    instructionName[(int)INS_TAX]="TAX";
    instructionName[(int)INS_TAY]="TAY";
    instructionName[(int)INS_TSX]="TSX";
    instructionName[(int)INS_TXA]="TXA";
    instructionName[(int)INS_TXS]="TXS";
    instructionName[(int)INS_TYA]="TYA";

    instructionName[(int)INS_ANC]="ANC";
    instructionName[(int)INS_ANE]="ANE";
    instructionName[(int)INS_ASR]="ASR";
    instructionName[(int)INS_ARR]="ARR";
    instructionName[(int)INS_DCP]="DCP";
    instructionName[(int)INS_ISC]="ISB";
    instructionName[(int)INS_ISC]="ISC";
    instructionName[(int)INS_LAS]="LAS";
    instructionName[(int)INS_LAX]="LAX";
    instructionName[(int)INS_LXA]="LXA";
    instructionName[(int)INS_RLA]="RLA";
    instructionName[(int)INS_RRA]="RRA";
    instructionName[(int)INS_SAX]="SAX";
    instructionName[(int)INS_SBX]="SBX";
    instructionName[(int)INS_SHA]="SHA";
    instructionName[(int)INS_SHS]="SHS";
    instructionName[(int)INS_SHY]="SHY";
    instructionName[(int)INS_SLO]="SLP";
    instructionName[(int)INS_SRE]="SRE";
    instructionName[(int)INS_NA] ="NA";
}

static void fillAllEntries()
{
    opdata[0x0].cycles=7;
    opdata[0x0].inst=INS_BRK;
    opdata[0x0].addrmode=ADR_IMP;
    opdata[0x1].cycles=6;
    opdata[0x1].inst=INS_ORA;
    opdata[0x1].addrmode=ADR_INDX;
    opdata[0x2].cycles=2;
    opdata[0x2].inst=INS_NA;
    opdata[0x2].addrmode=ADR_IMP;
    opdata[0x3].cycles=8;
    opdata[0x3].inst=INS_SLO;
    opdata[0x3].addrmode=ADR_INDX;
    opdata[0x4].cycles=3;
    opdata[0x4].inst=INS_NOP;
    opdata[0x4].addrmode=ADR_ZP;
    opdata[0x5].cycles=3;
    opdata[0x5].inst=INS_ORA;
    opdata[0x5].addrmode=ADR_ZP;
    opdata[0x6].cycles=5;
    opdata[0x6].inst=INS_ASL;
    opdata[0x6].addrmode=ADR_ZP;
    opdata[0x7].cycles=5;
    opdata[0x7].inst=INS_SLO;
    opdata[0x7].addrmode=ADR_ZP;
    opdata[0x8].cycles=3;
    opdata[0x8].inst=INS_PHP;
    opdata[0x8].addrmode=ADR_IMP;
    opdata[0x9].cycles=2;
    opdata[0x9].inst=INS_ORA;
    opdata[0x9].addrmode=ADR_IMM;
    opdata[0xA].cycles=2;
    opdata[0xA].inst=INS_ASLA;
    opdata[0xA].addrmode=ADR_IMP;
    opdata[0xB].cycles=2;
    opdata[0xB].inst=INS_ANC;
    opdata[0xB].addrmode=ADR_IMM;
    opdata[0xC].cycles=4;
    opdata[0xC].inst=INS_NOP;
    opdata[0xC].addrmode=ADR_ABS;
    opdata[0xD].cycles=4;
    opdata[0xD].inst=INS_ORA;
    opdata[0xD].addrmode=ADR_ABS;
    opdata[0xE].cycles=6;
    opdata[0xE].inst=INS_ASL;
    opdata[0xE].addrmode=ADR_ABS;
    opdata[0xF].cycles=6;
    opdata[0xF].inst=INS_SLO;
    opdata[0xF].addrmode=ADR_ABS;
    opdata[0x10].cycles=2;
    opdata[0x10].inst=INS_BPL;
    opdata[0x10].addrmode=ADR_REL;
    opdata[0x11].cycles=5;
    opdata[0x11].inst=INS_ORA;
    opdata[0x11].addrmode=ADR_INDY;
    opdata[0x12].cycles=2;
    opdata[0x12].inst=INS_NA;
    opdata[0x12].addrmode=ADR_INDZP;
    opdata[0x13].cycles=8;
    opdata[0x13].inst=INS_SLO;
    opdata[0x13].addrmode=ADR_INDY;
    opdata[0x14].cycles=4;
    opdata[0x14].inst=INS_NOP;
    opdata[0x14].addrmode=ADR_ZPX;
    opdata[0x15].cycles=4;
    opdata[0x15].inst=INS_ORA;
    opdata[0x15].addrmode=ADR_ZPX;
    opdata[0x16].cycles=6;
    opdata[0x16].inst=INS_ASL;
    opdata[0x16].addrmode=ADR_ZPX;
    opdata[0x17].cycles=6;
    opdata[0x17].inst=INS_SLO;
    opdata[0x17].addrmode=ADR_ZPX;
    opdata[0x18].cycles=2;
    opdata[0x18].inst=INS_CLC;
    opdata[0x18].addrmode=ADR_IMP;
    opdata[0x19].cycles=4;
    opdata[0x19].inst=INS_ORA;
    opdata[0x19].addrmode=ADR_ABSY;
    opdata[0x1A].cycles=2;
    opdata[0x1A].inst=INS_NOP;
    opdata[0x1A].addrmode=ADR_IMP;
    opdata[0x1B].cycles=7;
    opdata[0x1B].inst=INS_SLO;
    opdata[0x1B].addrmode=ADR_ABSY;
    opdata[0x1C].cycles=4;
    opdata[0x1C].inst=INS_NOP;
    opdata[0x1C].addrmode=ADR_ABSX;
    opdata[0x1D].cycles=4;
    opdata[0x1D].inst=INS_ORA;
    opdata[0x1D].addrmode=ADR_ABSX;
    opdata[0x1E].cycles=7;
    opdata[0x1E].inst=INS_ASL;
    opdata[0x1E].addrmode=ADR_ABSX;
    opdata[0x1F].cycles=7;
    opdata[0x1F].inst=INS_SLO;
    opdata[0x1F].addrmode=ADR_ABSX;
    opdata[0x20].cycles=6;
    opdata[0x20].inst=INS_JSR;
    opdata[0x20].addrmode=ADR_ABS;
    opdata[0x21].cycles=6;
    opdata[0x21].inst=INS_AND;
    opdata[0x21].addrmode=ADR_INDX;
    opdata[0x22].cycles=2;
    opdata[0x22].inst=INS_NA;
    opdata[0x22].addrmode=ADR_IMP;
    opdata[0x23].cycles=8;
    opdata[0x23].inst=INS_RLA;
    opdata[0x23].addrmode=ADR_INDX;
    opdata[0x24].cycles=3;
    opdata[0x24].inst=INS_BIT;
    opdata[0x24].addrmode=ADR_ZP;
    opdata[0x25].cycles=3;
    opdata[0x25].inst=INS_AND;
    opdata[0x25].addrmode=ADR_ZP;
    opdata[0x26].cycles=5;
    opdata[0x26].inst=INS_ROL;
    opdata[0x26].addrmode=ADR_ZP;
    opdata[0x27].cycles=5;
    opdata[0x27].inst=INS_RLA;
    opdata[0x27].addrmode=ADR_ZP;
    opdata[0x28].cycles=4;
    opdata[0x28].inst=INS_PLP;
    opdata[0x28].addrmode=ADR_IMP;
    opdata[0x29].cycles=2;
    opdata[0x29].inst=INS_AND;
    opdata[0x29].addrmode=ADR_IMM;
    opdata[0x2A].cycles=2;
    opdata[0x2A].inst=INS_ROLA;
    opdata[0x2A].addrmode=ADR_IMP;
    opdata[0x2B].cycles=2;
    opdata[0x2B].inst=INS_ANC;
    opdata[0x2B].addrmode=ADR_IMM;
    opdata[0x2C].cycles=4;
    opdata[0x2C].inst=INS_BIT;
    opdata[0x2C].addrmode=ADR_ABS;
    opdata[0x2D].cycles=4;
    opdata[0x2D].inst=INS_AND;
    opdata[0x2D].addrmode=ADR_ABS;
    opdata[0x2E].cycles=6;
    opdata[0x2E].inst=INS_ROL;
    opdata[0x2E].addrmode=ADR_ABS;
    opdata[0x2F].cycles=6;
    opdata[0x2F].inst=INS_RLA;
    opdata[0x2F].addrmode=ADR_ABS;
    opdata[0x30].cycles=2;
    opdata[0x30].inst=INS_BMI;
    opdata[0x30].addrmode=ADR_REL;
    opdata[0x31].cycles=5;
    opdata[0x31].inst=INS_AND;
    opdata[0x31].addrmode=ADR_INDY;
    opdata[0x32].cycles=2;
    opdata[0x32].inst=INS_NA;
    opdata[0x32].addrmode=ADR_INDZP;
    opdata[0x33].cycles=8;
    opdata[0x33].inst=INS_RLA;
    opdata[0x33].addrmode=ADR_INDY;
    opdata[0x34].cycles=4;
    opdata[0x34].inst=INS_NOP;
    opdata[0x34].addrmode=ADR_ZPX;
    opdata[0x35].cycles=4;
    opdata[0x35].inst=INS_AND;
    opdata[0x35].addrmode=ADR_ZPX;
    opdata[0x36].cycles=6;
    opdata[0x36].inst=INS_ROL;
    opdata[0x36].addrmode=ADR_ZPX;
    opdata[0x37].cycles=6;
    opdata[0x37].inst=INS_RLA;
    opdata[0x37].addrmode=ADR_ZPX;
    opdata[0x38].cycles=2;
    opdata[0x38].inst=INS_SEC;
    opdata[0x38].addrmode=ADR_IMP;
    opdata[0x39].cycles=4;
    opdata[0x39].inst=INS_AND;
    opdata[0x39].addrmode=ADR_ABSY;
    opdata[0x3A].cycles=2;
    opdata[0x3A].inst=INS_NOP;
    opdata[0x3A].addrmode=ADR_IMP;
    opdata[0x3B].cycles=7;
    opdata[0x3B].inst=INS_RLA;
    opdata[0x3B].addrmode=ADR_ABSY;
    opdata[0x3C].cycles=4;
    opdata[0x3C].inst=INS_NOP;
    opdata[0x3C].addrmode=ADR_ABSX;
    opdata[0x3D].cycles=4;
    opdata[0x3D].inst=INS_AND;
    opdata[0x3D].addrmode=ADR_ABSX;
    opdata[0x3E].cycles=7;
    opdata[0x3E].inst=INS_ROL;
    opdata[0x3E].addrmode=ADR_ABSX;
    opdata[0x3F].cycles=7;
    opdata[0x3F].inst=INS_RLA;
    opdata[0x3F].addrmode=ADR_ABSX;
    opdata[0x40].cycles=6;
    opdata[0x40].inst=INS_RTI;
    opdata[0x40].addrmode=ADR_IMP;
    opdata[0x41].cycles=6;
    opdata[0x41].inst=INS_EOR;
    opdata[0x41].addrmode=ADR_INDX;
    opdata[0x42].cycles=2;
    opdata[0x42].inst=INS_NA;
    opdata[0x42].addrmode=ADR_IMP;
    opdata[0x43].cycles=8;
    opdata[0x43].inst=INS_SRE;
    opdata[0x43].addrmode=ADR_INDX;
    opdata[0x44].cycles=3;
    opdata[0x44].inst=INS_NOP;
    opdata[0x44].addrmode=ADR_ZP;
    opdata[0x45].cycles=3;
    opdata[0x45].inst=INS_EOR;
    opdata[0x45].addrmode=ADR_ZP;
    opdata[0x46].cycles=5;
    opdata[0x46].inst=INS_LSR;
    opdata[0x46].addrmode=ADR_ZP;
    opdata[0x47].cycles=5;
    opdata[0x47].inst=INS_SRE;
    opdata[0x47].addrmode=ADR_ZP;
    opdata[0x48].cycles=3;
    opdata[0x48].inst=INS_PHA;
    opdata[0x48].addrmode=ADR_IMP;
    opdata[0x49].cycles=2;
    opdata[0x49].inst=INS_EOR;
    opdata[0x49].addrmode=ADR_IMM;
    opdata[0x4A].cycles=2;
    opdata[0x4A].inst=INS_LSRA;
    opdata[0x4A].addrmode=ADR_IMP;
    opdata[0x4B].cycles=2;
    opdata[0x4B].inst=INS_ASR;
    opdata[0x4B].addrmode=ADR_IMM;
    opdata[0x4C].cycles=3;
    opdata[0x4C].inst=INS_JMP;
    opdata[0x4C].addrmode=ADR_ABS;
    opdata[0x4D].cycles=4;
    opdata[0x4D].inst=INS_EOR;
    opdata[0x4D].addrmode=ADR_ABS;
    opdata[0x4E].cycles=6;
    opdata[0x4E].inst=INS_LSR;
    opdata[0x4E].addrmode=ADR_ABS;
    opdata[0x4F].cycles=6;
    opdata[0x4F].inst=INS_SRE;
    opdata[0x4F].addrmode=ADR_ABS;
    opdata[0x50].cycles=2;
    opdata[0x50].inst=INS_BVC;
    opdata[0x50].addrmode=ADR_REL;
    opdata[0x51].cycles=5;
    opdata[0x51].inst=INS_EOR;
    opdata[0x51].addrmode=ADR_INDY;
    opdata[0x52].cycles=2;
    opdata[0x52].inst=INS_NA;
    opdata[0x52].addrmode=ADR_INDZP;
    opdata[0x53].cycles=8;
    opdata[0x53].inst=INS_SRE;
    opdata[0x53].addrmode=ADR_INDY;
    opdata[0x54].cycles=4;
    opdata[0x54].inst=INS_NOP;
    opdata[0x54].addrmode=ADR_ZPX;
    opdata[0x55].cycles=4;
    opdata[0x55].inst=INS_EOR;
    opdata[0x55].addrmode=ADR_ZPX;
    opdata[0x56].cycles=6;
    opdata[0x56].inst=INS_LSR;
    opdata[0x56].addrmode=ADR_ZPX;
    opdata[0x57].cycles=6;
    opdata[0x57].inst=INS_SRE;
    opdata[0x57].addrmode=ADR_ZPX;
    opdata[0x58].cycles=2;
    opdata[0x58].inst=INS_CLI;
    opdata[0x58].addrmode=ADR_IMP;
    opdata[0x59].cycles=4;
    opdata[0x59].inst=INS_EOR;
    opdata[0x59].addrmode=ADR_ABSY;
    opdata[0x5A].cycles=2;
    opdata[0x5A].inst=INS_NOP;
    opdata[0x5A].addrmode=ADR_IMP;
    opdata[0x5B].cycles=7;
    opdata[0x5B].inst=INS_SRE;
    opdata[0x5B].addrmode=ADR_ABSY;
    opdata[0x5C].cycles=4;
    opdata[0x5C].inst=INS_NOP;
    opdata[0x5C].addrmode=ADR_ABSX;
    opdata[0x5D].cycles=4;
    opdata[0x5D].inst=INS_EOR;
    opdata[0x5D].addrmode=ADR_ABSX;
    opdata[0x5E].cycles=7;
    opdata[0x5E].inst=INS_LSR;
    opdata[0x5E].addrmode=ADR_ABSX;
    opdata[0x5F].cycles=7;
    opdata[0x5F].inst=INS_SRE;
    opdata[0x5F].addrmode=ADR_ABSX;
    opdata[0x60].cycles=6;
    opdata[0x60].inst=INS_RTS;
    opdata[0x60].addrmode=ADR_IMP;
    opdata[0x61].cycles=6;
    opdata[0x61].inst=INS_ADC;
    opdata[0x61].addrmode=ADR_INDX;
    opdata[0x62].cycles=2;
    opdata[0x62].inst=INS_NA;
    opdata[0x62].addrmode=ADR_IMP;
    opdata[0x63].cycles=8;
    opdata[0x63].inst=INS_RRA;
    opdata[0x63].addrmode=ADR_INDX;
    opdata[0x64].cycles=3;
    opdata[0x64].inst=INS_NOP;
    opdata[0x64].addrmode=ADR_ZP;
    opdata[0x65].cycles=3;
    opdata[0x65].inst=INS_ADC;
    opdata[0x65].addrmode=ADR_ZP;
    opdata[0x66].cycles=5;
    opdata[0x66].inst=INS_ROR;
    opdata[0x66].addrmode=ADR_ZP;
    opdata[0x67].cycles=5;
    opdata[0x67].inst=INS_RRA;
    opdata[0x67].addrmode=ADR_ZP;
    opdata[0x68].cycles=4;
    opdata[0x68].inst=INS_PLA;
    opdata[0x68].addrmode=ADR_IMP;
    opdata[0x69].cycles=2;
    opdata[0x69].inst=INS_ADC;
    opdata[0x69].addrmode=ADR_IMM;
    opdata[0x6A].cycles=2;
    opdata[0x6A].inst=INS_RORA;
    opdata[0x6A].addrmode=ADR_IMP;
    opdata[0x6B].cycles=2;
    opdata[0x6B].inst=INS_ARR;
    opdata[0x6B].addrmode=ADR_IMM;
    opdata[0x6C].cycles=5;
    opdata[0x6C].inst=INS_JMP;
    opdata[0x6C].addrmode=ADR_IND;
    opdata[0x6D].cycles=4;
    opdata[0x6D].inst=INS_ADC;
    opdata[0x6D].addrmode=ADR_ABS;
    opdata[0x6E].cycles=6;
    opdata[0x6E].inst=INS_ROR;
    opdata[0x6E].addrmode=ADR_ABS;
    opdata[0x6F].cycles=6;
    opdata[0x6F].inst=INS_RRA;
    opdata[0x6F].addrmode=ADR_ABS;
    opdata[0x70].cycles=2;
    opdata[0x70].inst=INS_BVS;
    opdata[0x70].addrmode=ADR_REL;
    opdata[0x71].cycles=5;
    opdata[0x71].inst=INS_ADC;
    opdata[0x71].addrmode=ADR_INDY;
    opdata[0x72].cycles=2;
    opdata[0x72].inst=INS_NA;
    opdata[0x72].addrmode=ADR_INDZP;
    opdata[0x73].cycles=8;
    opdata[0x73].inst=INS_RRA;
    opdata[0x73].addrmode=ADR_INDY;
    opdata[0x74].cycles=4;
    opdata[0x74].inst=INS_NOP;
    opdata[0x74].addrmode=ADR_ZPX;
    opdata[0x75].cycles=4;
    opdata[0x75].inst=INS_ADC;
    opdata[0x75].addrmode=ADR_ZPX;
    opdata[0x76].cycles=6;
    opdata[0x76].inst=INS_ROR;
    opdata[0x76].addrmode=ADR_ZPX;
    opdata[0x77].cycles=6;
    opdata[0x77].inst=INS_RRA;
    opdata[0x77].addrmode=ADR_ZPX;
    opdata[0x78].cycles=2;
    opdata[0x78].inst=INS_SEI;
    opdata[0x78].addrmode=ADR_IMP;
    opdata[0x79].cycles=4;
    opdata[0x79].inst=INS_ADC;
    opdata[0x79].addrmode=ADR_ABSY;
    opdata[0x7A].cycles=2;
    opdata[0x7A].inst=INS_NOP;
    opdata[0x7A].addrmode=ADR_IMP;
    opdata[0x7B].cycles=7;
    opdata[0x7B].inst=INS_RRA;
    opdata[0x7B].addrmode=ADR_ABSY;
    opdata[0x7C].cycles=4;
    opdata[0x7C].inst=INS_NOP;
    opdata[0x7C].addrmode=ADR_ABSX;
    opdata[0x7D].cycles=4;
    opdata[0x7D].inst=INS_ADC;
    opdata[0x7D].addrmode=ADR_ABSX;
    opdata[0x7E].cycles=7;
    opdata[0x7E].inst=INS_ROR;
    opdata[0x7E].addrmode=ADR_ABSX;
    opdata[0x7F].cycles=7;
    opdata[0x7F].inst=INS_RRA;
    opdata[0x7F].addrmode=ADR_ABSX;
    opdata[0x80].cycles=2;
    opdata[0x80].inst=INS_NOP;
    opdata[0x80].addrmode=ADR_IMM;
    opdata[0x81].cycles=6;
    opdata[0x81].inst=INS_STA;
    opdata[0x81].addrmode=ADR_INDX;
    opdata[0x82].cycles=2;
    opdata[0x82].inst=INS_NOP;
    opdata[0x82].addrmode=ADR_IMM;
    opdata[0x83].cycles=6;
    opdata[0x83].inst=INS_SAX;
    opdata[0x83].addrmode=ADR_INDX;
    opdata[0x84].cycles=3;
    opdata[0x84].inst=INS_STY;
    opdata[0x84].addrmode=ADR_ZP;
    opdata[0x85].cycles=3;
    opdata[0x85].inst=INS_STA;
    opdata[0x85].addrmode=ADR_ZP;
    opdata[0x86].cycles=3;
    opdata[0x86].inst=INS_STX;
    opdata[0x86].addrmode=ADR_ZP;
    opdata[0x87].cycles=3;
    opdata[0x87].inst=INS_SAX;
    opdata[0x87].addrmode=ADR_ZP;
    opdata[0x88].cycles=2;
    opdata[0x88].inst=INS_DEY;
    opdata[0x88].addrmode=ADR_IMP;
    opdata[0x89].cycles=2;
    opdata[0x89].inst=INS_NOP;
    opdata[0x89].addrmode=ADR_IMM;
    opdata[0x8A].cycles=2;
    opdata[0x8A].inst=INS_TXA;
    opdata[0x8A].addrmode=ADR_IMP;
    opdata[0x8B].cycles=2;
    opdata[0x8B].inst=INS_ANE;
    opdata[0x8B].addrmode=ADR_IMM;
    opdata[0x8C].cycles=4;
    opdata[0x8C].inst=INS_STY;
    opdata[0x8C].addrmode=ADR_ABS;
    opdata[0x8D].cycles=4;
    opdata[0x8D].inst=INS_STA;
    opdata[0x8D].addrmode=ADR_ABS;
    opdata[0x8E].cycles=4;
    opdata[0x8E].inst=INS_STX;
    opdata[0x8E].addrmode=ADR_ABS;
    opdata[0x8F].cycles=4;
    opdata[0x8F].inst=INS_SAX;
    opdata[0x8F].addrmode=ADR_ABS;
    opdata[0x90].cycles=2;
    opdata[0x90].inst=INS_BCC;
    opdata[0x90].addrmode=ADR_REL;
    opdata[0x91].cycles=6;
    opdata[0x91].inst=INS_STA;
    opdata[0x91].addrmode=ADR_INDY;
    opdata[0x92].cycles=2;
    opdata[0x92].inst=INS_NA;
    opdata[0x92].addrmode=ADR_INDZP;
    opdata[0x93].cycles=6;
    opdata[0x93].inst=INS_SHA;
    opdata[0x93].addrmode=ADR_INDY;
    opdata[0x94].cycles=4;
    opdata[0x94].inst=INS_STY;
    opdata[0x94].addrmode=ADR_ZPX;
    opdata[0x95].cycles=4;
    opdata[0x95].inst=INS_STA;
    opdata[0x95].addrmode=ADR_ZPX;
    opdata[0x96].cycles=4;
    opdata[0x96].inst=INS_STX;
    opdata[0x96].addrmode=ADR_ZPY;
    opdata[0x97].cycles=4;
    opdata[0x97].inst=INS_SAX;
    opdata[0x97].addrmode=ADR_ZPY;
    opdata[0x98].cycles=2;
    opdata[0x98].inst=INS_TYA;
    opdata[0x98].addrmode=ADR_IMP;
    opdata[0x99].cycles=5;
    opdata[0x99].inst=INS_STA;
    opdata[0x99].addrmode=ADR_ABSY;
    opdata[0x9A].cycles=2;
    opdata[0x9A].inst=INS_TXS;
    opdata[0x9A].addrmode=ADR_IMP;
    opdata[0x9B].cycles=5;
    opdata[0x9B].inst=INS_SHS;
    opdata[0x9B].addrmode=ADR_ABSY;
    opdata[0x9C].cycles=5;
    opdata[0x9C].inst=INS_SHY;
    opdata[0x9C].addrmode=ADR_ABSX;
    opdata[0x9D].cycles=5;
    opdata[0x9D].inst=INS_STA;
    opdata[0x9D].addrmode=ADR_ABSX;
    opdata[0x9E].cycles=5;
    opdata[0x9E].inst=INS_SHX;
    opdata[0x9E].addrmode=ADR_ABSY;
    opdata[0x9F].cycles=5;
    opdata[0x9F].inst=INS_SHA;
    opdata[0x9F].addrmode=ADR_ABSY;
    opdata[0xA0].cycles=2/*3*/;
    opdata[0xA0].inst=INS_LDY;
    opdata[0xA0].addrmode=ADR_IMM;
    opdata[0xA1].cycles=6;
    opdata[0xA1].inst=INS_LDA;
    opdata[0xA1].addrmode=ADR_INDX;
    opdata[0xA2].cycles=2/*3*/;
    opdata[0xA2].inst=INS_LDX;
    opdata[0xA2].addrmode=ADR_IMM;
    opdata[0xA3].cycles=6;
    opdata[0xA3].inst=INS_LAX;
    opdata[0xA3].addrmode=ADR_INDX;
    opdata[0xA4].cycles=3;
    opdata[0xA4].inst=INS_LDY;
    opdata[0xA4].addrmode=ADR_ZP;
    opdata[0xA5].cycles=3;
    opdata[0xA5].inst=INS_LDA;
    opdata[0xA5].addrmode=ADR_ZP;
    opdata[0xA6].cycles=3;
    opdata[0xA6].inst=INS_LDX;
    opdata[0xA6].addrmode=ADR_ZP;
    opdata[0xA7].cycles=4;
    opdata[0xA7].inst=INS_LAX;
    opdata[0xA7].addrmode=ADR_ZP;
    opdata[0xA8].cycles=2;
    opdata[0xA8].inst=INS_TAY;
    opdata[0xA8].addrmode=ADR_IMP;
    opdata[0xA9].cycles=2;
    opdata[0xA9].inst=INS_LDA;
    opdata[0xA9].addrmode=ADR_IMM;
    opdata[0xAA].cycles=2;
    opdata[0xAA].inst=INS_TAX;
    opdata[0xAA].addrmode=ADR_IMP;
    opdata[0xAB].cycles=2;
    opdata[0xAB].inst=INS_LXA;
    opdata[0xAB].addrmode=ADR_IMM;
    opdata[0xAC].cycles=4;
    opdata[0xAC].inst=INS_LDY;
    opdata[0xAC].addrmode=ADR_ABS;
    opdata[0xAD].cycles=4;
    opdata[0xAD].inst=INS_LDA;
    opdata[0xAD].addrmode=ADR_ABS;
    opdata[0xAE].cycles=4;
    opdata[0xAE].inst=INS_LDX;
    opdata[0xAE].addrmode=ADR_ABS;
    opdata[0xAF].cycles=4;
    opdata[0xAF].inst=INS_LAX;
    opdata[0xAF].addrmode=ADR_ABS;
    opdata[0xB0].cycles=2;
    opdata[0xB0].inst=INS_BCS;
    opdata[0xB0].addrmode=ADR_REL;
    opdata[0xB1].cycles=5;
    opdata[0xB1].inst=INS_LDA;
    opdata[0xB1].addrmode=ADR_INDY;
    opdata[0xB2].cycles=2;
    opdata[0xB2].inst=INS_NA;
    opdata[0xB2].addrmode=ADR_INDZP;
    opdata[0xB3].cycles=5;
    opdata[0xB3].inst=INS_LAX;
    opdata[0xB3].addrmode=ADR_INDY;
    opdata[0xB4].cycles=4;
    opdata[0xB4].inst=INS_LDY;
    opdata[0xB4].addrmode=ADR_ZPX;
    opdata[0xB5].cycles=4;
    opdata[0xB5].inst=INS_LDA;
    opdata[0xB5].addrmode=ADR_ZPX;
    opdata[0xB6].cycles=4;
    opdata[0xB6].inst=INS_LDX;
    opdata[0xB6].addrmode=ADR_ZPY;
    opdata[0xB7].cycles=4;
    opdata[0xB7].inst=INS_LAX;
    opdata[0xB7].addrmode=ADR_ZPY;
    opdata[0xB8].cycles=2;
    opdata[0xB8].inst=INS_CLV;
    opdata[0xB8].addrmode=ADR_IMP;
    opdata[0xB9].cycles=4;
    opdata[0xB9].inst=INS_LDA;
    opdata[0xB9].addrmode=ADR_ABSY;
    opdata[0xBA].cycles=2;
    opdata[0xBA].inst=INS_TSX;
    opdata[0xBA].addrmode=ADR_IMP;
    opdata[0xBB].cycles=4;
    opdata[0xBB].inst=INS_LAS;
    opdata[0xBB].addrmode=ADR_ABSY;
    opdata[0xBC].cycles=4;
    opdata[0xBC].inst=INS_LDY;
    opdata[0xBC].addrmode=ADR_ABSX;
    opdata[0xBD].cycles=4;
    opdata[0xBD].inst=INS_LDA;
    opdata[0xBD].addrmode=ADR_ABSX;
    opdata[0xBE].cycles=4;
    opdata[0xBE].inst=INS_LDX;
    opdata[0xBE].addrmode=ADR_ABSY;
    opdata[0xBF].cycles=4;
    opdata[0xBF].inst=INS_LAX;
    opdata[0xBF].addrmode=ADR_ABSY;
    opdata[0xC0].cycles=2;
    opdata[0xC0].inst=INS_CPY;
    opdata[0xC0].addrmode=ADR_IMM;
    opdata[0xC1].cycles=6;
    opdata[0xC1].inst=INS_CMP;
    opdata[0xC1].addrmode=ADR_INDX;
    opdata[0xC2].cycles=2;
    opdata[0xC2].inst=INS_NOP;
    opdata[0xC2].addrmode=ADR_IMM;
    opdata[0xC3].cycles=8;
    opdata[0xC3].inst=INS_DCP;
    opdata[0xC3].addrmode=ADR_INDX;
    opdata[0xC4].cycles=3;
    opdata[0xC4].inst=INS_CPY;
    opdata[0xC4].addrmode=ADR_ZP;
    opdata[0xC5].cycles=3;
    opdata[0xC5].inst=INS_CMP;
    opdata[0xC5].addrmode=ADR_ZP;
    opdata[0xC6].cycles=5;
    opdata[0xC6].inst=INS_DEC;
    opdata[0xC6].addrmode=ADR_ZP;
    opdata[0xC7].cycles=5;
    opdata[0xC7].inst=INS_DCP;
    opdata[0xC7].addrmode=ADR_ZP;
    opdata[0xC8].cycles=2;
    opdata[0xC8].inst=INS_INY;
    opdata[0xC8].addrmode=ADR_IMP;
    opdata[0xC9].cycles=2;
    opdata[0xC9].inst=INS_CMP;
    opdata[0xC9].addrmode=ADR_IMM;
    opdata[0xCA].cycles=2;
    opdata[0xCA].inst=INS_DEX;
    opdata[0xCA].addrmode=ADR_IMP;
    opdata[0xCB].cycles=2;
    opdata[0xCB].inst=INS_SBX;
    opdata[0xCB].addrmode=ADR_IMM;
    opdata[0xCC].cycles=4;
    opdata[0xCC].inst=INS_CPY;
    opdata[0xCC].addrmode=ADR_ABS;
    opdata[0xCD].cycles=4;
    opdata[0xCD].inst=INS_CMP;
    opdata[0xCD].addrmode=ADR_ABS;
    opdata[0xCE].cycles=6;
    opdata[0xCE].inst=INS_DEC;
    opdata[0xCE].addrmode=ADR_ABS;
    opdata[0xCF].cycles=6;
    opdata[0xCF].inst=INS_DCP;
    opdata[0xCF].addrmode=ADR_ABS;
    opdata[0xD0].cycles=2;
    opdata[0xD0].inst=INS_BNE;
    opdata[0xD0].addrmode=ADR_REL;
    opdata[0xD1].cycles=5;
    opdata[0xD1].inst=INS_CMP;
    opdata[0xD1].addrmode=ADR_INDY;
    opdata[0xD2].cycles=2;
    opdata[0xD2].inst=INS_NA;
    opdata[0xD2].addrmode=ADR_INDZP;
    opdata[0xD3].cycles=8;
    opdata[0xD3].inst=INS_DCP;
    opdata[0xD3].addrmode=ADR_INDY;
    opdata[0xD4].cycles=4;
    opdata[0xD4].inst=INS_NOP;
    opdata[0xD4].addrmode=ADR_ZPX;
    opdata[0xD5].cycles=4;
    opdata[0xD5].inst=INS_CMP;
    opdata[0xD5].addrmode=ADR_ZPX;
    opdata[0xD6].cycles=6;
    opdata[0xD6].inst=INS_DEC;
    opdata[0xD6].addrmode=ADR_ZPX;
    opdata[0xD7].cycles=6;
    opdata[0xD7].inst=INS_DCP;
    opdata[0xD7].addrmode=ADR_ZPX;
    opdata[0xD8].cycles=2;
    opdata[0xD8].inst=INS_CLD;
    opdata[0xD8].addrmode=ADR_IMP;
    opdata[0xD9].cycles=4;
    opdata[0xD9].inst=INS_CMP;
    opdata[0xD9].addrmode=ADR_ABSY;
    opdata[0xDA].cycles=2;
    opdata[0xDA].inst=INS_NOP;
    opdata[0xDA].addrmode=ADR_IMP;
    opdata[0xDB].cycles=7;
    opdata[0xDB].inst=INS_DCP;
    opdata[0xDB].addrmode=ADR_ABSY;
    opdata[0xDC].cycles=4;
    opdata[0xDC].inst=INS_NOP;
    opdata[0xDC].addrmode=ADR_ABSX;
    opdata[0xDD].cycles=4;
    opdata[0xDD].inst=INS_CMP;
    opdata[0xDD].addrmode=ADR_ABSX;
    opdata[0xDE].cycles=7;
    opdata[0xDE].inst=INS_DEC;
    opdata[0xDE].addrmode=ADR_ABSX;
    opdata[0xDF].cycles=7;
    opdata[0xDF].inst=INS_DCP;
    opdata[0xDF].addrmode=ADR_ABSX;
    opdata[0xE0].cycles=2;
    opdata[0xE0].inst=INS_CPX;
    opdata[0xE0].addrmode=ADR_IMM;
    opdata[0xE1].cycles=6;
    opdata[0xE1].inst=INS_SBC;
    opdata[0xE1].addrmode=ADR_INDX;
    opdata[0xE2].cycles=2;
    opdata[0xE2].inst=INS_NOP;
    opdata[0xE2].addrmode=ADR_IMM;
    opdata[0xE3].cycles=8;
    opdata[0xE3].inst=INS_ISB;
    opdata[0xE3].addrmode=ADR_INDX;
    opdata[0xE4].cycles=3;
    opdata[0xE4].inst=INS_CPX;
    opdata[0xE4].addrmode=ADR_ZP;
    opdata[0xE5].cycles=3;
    opdata[0xE5].inst=INS_SBC;
    opdata[0xE5].addrmode=ADR_ZP;
    opdata[0xE6].cycles=5;
    opdata[0xE6].inst=INS_INC;
    opdata[0xE6].addrmode=ADR_ZP;
    opdata[0xE7].cycles=5;
    opdata[0xE7].inst=INS_ISB;
    opdata[0xE7].addrmode=ADR_ZP;
    opdata[0xE8].cycles=2;
    opdata[0xE8].inst=INS_INX;
    opdata[0xE8].addrmode=ADR_IMP;
    opdata[0xE9].cycles=2;
    opdata[0xE9].inst=INS_SBC;
    opdata[0xE9].addrmode=ADR_IMM;
    opdata[0xEA].cycles=2;
    opdata[0xEA].inst=INS_NOP;
    opdata[0xEA].addrmode=ADR_IMP;
    opdata[0xEB].cycles=2;
    opdata[0xEB].inst=INS_SBC;
    opdata[0xEB].addrmode=ADR_IMM;
    opdata[0xEC].cycles=4;
    opdata[0xEC].inst=INS_CPX;
    opdata[0xEC].addrmode=ADR_ABS;
    opdata[0xED].cycles=4;
    opdata[0xED].inst=INS_SBC;
    opdata[0xED].addrmode=ADR_ABS;
    opdata[0xEE].cycles=6;
    opdata[0xEE].inst=INS_INC;
    opdata[0xEE].addrmode=ADR_ABS;
    opdata[0xEF].cycles=6;
    opdata[0xEF].inst=INS_ISB;
    opdata[0xEF].addrmode=ADR_ABS;
    opdata[0xF0].cycles=2;
    opdata[0xF0].inst=INS_BEQ;
    opdata[0xF0].addrmode=ADR_REL;
    opdata[0xF1].cycles=5;
    opdata[0xF1].inst=INS_SBC;
    opdata[0xF1].addrmode=ADR_INDY;
    opdata[0xF2].cycles=2;
    opdata[0xF2].inst=INS_NA;
    opdata[0xF2].addrmode=ADR_INDZP;
    opdata[0xF3].cycles=8;
    opdata[0xF3].inst=INS_ISB;
    opdata[0xF3].addrmode=ADR_INDY;
    opdata[0xF4].cycles=4;
    opdata[0xF4].inst=INS_NOP;
    opdata[0xF4].addrmode=ADR_ZPX;
    opdata[0xF5].cycles=4;
    opdata[0xF5].inst=INS_SBC;
    opdata[0xF5].addrmode=ADR_ZPX;
    opdata[0xF6].cycles=6;
    opdata[0xF6].inst=INS_INC;
    opdata[0xF6].addrmode=ADR_ZPX;
    opdata[0xF7].cycles=6;
    opdata[0xF7].inst=INS_ISB;
    opdata[0xF7].addrmode=ADR_ZPX;
    opdata[0xF8].cycles=2;
    opdata[0xF8].inst=INS_SED;
    opdata[0xF8].addrmode=ADR_IMP;
    opdata[0xF9].cycles=4;
    opdata[0xF9].inst=INS_SBC;
    opdata[0xF9].addrmode=ADR_ABSY;
    opdata[0xFA].cycles=2;
    opdata[0xFA].inst=INS_NOP;
    opdata[0xFA].addrmode=ADR_IMP;
    opdata[0xFB].cycles=7;
    opdata[0xFB].inst=INS_ISB;
    opdata[0xFB].addrmode=ADR_ABSY;
    opdata[0xFC].cycles=4;
    opdata[0xFC].inst=INS_NOP;
    opdata[0xFC].addrmode=ADR_ABSX;
    opdata[0xFD].cycles=4;
    opdata[0xFD].inst=INS_SBC;
    opdata[0xFD].addrmode=ADR_ABSX;
    opdata[0xFE].cycles=7;
    opdata[0xFE].inst=INS_INC;
    opdata[0xFE].addrmode=ADR_ABSX;
    opdata[0xFF].cycles=7;
    opdata[0xFF].inst=INS_ISB;
    opdata[0xFF].addrmode=ADR_ABSX;
}

static bool regOp(const M6502_INST inst, const opcode_t opcode, const M6502_ADDRMODE adrmode, uint8_t size)
{
    // make sure "inst" is consistent
    assert(opdata[opcode].inst == inst);

    // handle conflicts of other info
    if (adrmode != opdata[opcode].addrmode)
    {
        printf("[X] Error: Data Mismatch\n");
        printf("\t opcode: 0x%02x(%s)\n", opcode, opcode::instName(opcode));
        if (inst != opdata[opcode].inst)
        {
            printf("\tinst=%s[%u] (ref=%s[%u])\n", opcode::instName(inst), inst, opcode::instName(opdata[opcode].inst), (M6502_INST)opdata[opcode].inst);
        }
        if (adrmode != opdata[opcode].addrmode)
        {
            printf("\taddrmode=%u (ref=%u)\n", adrmode, (M6502_ADDRMODE)opdata[opcode].addrmode);
        }
        return false;
    }

    usualOp[opcode]=true;
    opdata[opcode].size=size;
    return true;
}

static void registerCommonOpcodes()
{
    // ADC:
    regOp(INS_ADC,0x69,ADR_IMM,2);
    regOp(INS_ADC,0x65,ADR_ZP,2);
    regOp(INS_ADC,0x75,ADR_ZPX,2);
    regOp(INS_ADC,0x6D,ADR_ABS,3);
    regOp(INS_ADC,0x7D,ADR_ABSX,3);
    regOp(INS_ADC,0x79,ADR_ABSY,3);
    regOp(INS_ADC,0x61,ADR_PREIDXIND,2);
    regOp(INS_ADC,0x71,ADR_POSTIDXIND,2);

    // AND:
    regOp(INS_AND,0x29,ADR_IMM,2);
    regOp(INS_AND,0x25,ADR_ZP,2);
    regOp(INS_AND,0x35,ADR_ZPX,2);
    regOp(INS_AND,0x2D,ADR_ABS,3);
    regOp(INS_AND,0x3D,ADR_ABSX,3);
    regOp(INS_AND,0x39,ADR_ABSY,3);
    regOp(INS_AND,0x21,ADR_PREIDXIND,2);
    regOp(INS_AND,0x31,ADR_POSTIDXIND,2);

    // ASL:
    regOp(INS_ASLA,0x0A,ADR_ACC,1);
    regOp(INS_ASL,0x06,ADR_ZP,2);
    regOp(INS_ASL,0x16,ADR_ZPX,2);
    regOp(INS_ASL,0x0E,ADR_ABS,3);
    regOp(INS_ASL,0x1E,ADR_ABSX,3);

    // BCC:
    regOp(INS_BCC,0x90,ADR_REL,2);

    // BCS:
    regOp(INS_BCS,0xB0,ADR_REL,2);

    // BEQ:
    regOp(INS_BEQ,0xF0,ADR_REL,2);

    // BIT:
    regOp(INS_BIT,0x24,ADR_ZP,2);
    regOp(INS_BIT,0x2C,ADR_ABS,3);

    // BMI:
    regOp(INS_BMI,0x30,ADR_REL,2);

    // BNE:
    regOp(INS_BNE,0xD0,ADR_REL,2);

    // BPL:
    regOp(INS_BPL,0x10,ADR_REL,2);

    // BRK:
    regOp(INS_BRK,0x00,ADR_IMP,1);

    // BVC:
    regOp(INS_BVC,0x50,ADR_REL,2);

    // BVS:
    regOp(INS_BVS,0x70,ADR_REL,2);

    // CLC:
    regOp(INS_CLC,0x18,ADR_IMP,1);

    // CLD:
    regOp(INS_CLD,0xD8,ADR_IMP,1);

    // CLI:
    regOp(INS_CLI,0x58,ADR_IMP,1);

    // CLV:
    regOp(INS_CLV,0xB8,ADR_IMP,1);

    // CMP:
    regOp(INS_CMP,0xC9,ADR_IMM,2);
    regOp(INS_CMP,0xC5,ADR_ZP,2);
    regOp(INS_CMP,0xD5,ADR_ZPX,2);
    regOp(INS_CMP,0xCD,ADR_ABS,3);
    regOp(INS_CMP,0xDD,ADR_ABSX,3);
    regOp(INS_CMP,0xD9,ADR_ABSY,3);
    regOp(INS_CMP,0xC1,ADR_PREIDXIND,2);
    regOp(INS_CMP,0xD1,ADR_POSTIDXIND,2);

    // CPX:
    regOp(INS_CPX,0xE0,ADR_IMM,2);
    regOp(INS_CPX,0xE4,ADR_ZP,2);
    regOp(INS_CPX,0xEC,ADR_ABS,3);

    // CPY:
    regOp(INS_CPY,0xC0,ADR_IMM,2);
    regOp(INS_CPY,0xC4,ADR_ZP,2);
    regOp(INS_CPY,0xCC,ADR_ABS,3);

    // DEC:
    regOp(INS_DEC,0xC6,ADR_ZP,2);
    regOp(INS_DEC,0xD6,ADR_ZPX,2);
    regOp(INS_DEC,0xCE,ADR_ABS,3);
    regOp(INS_DEC,0xDE,ADR_ABSX,3);

    // DEX:
    regOp(INS_DEX,0xCA,ADR_IMP,1);

    // DEY:
    regOp(INS_DEY,0x88,ADR_IMP,1);

    // EOR:
    regOp(INS_EOR,0x49,ADR_IMM,2);
    regOp(INS_EOR,0x45,ADR_ZP,2);
    regOp(INS_EOR,0x55,ADR_ZPX,2);
    regOp(INS_EOR,0x4D,ADR_ABS,3);
    regOp(INS_EOR,0x5D,ADR_ABSX,3);
    regOp(INS_EOR,0x59,ADR_ABSY,3);
    regOp(INS_EOR,0x41,ADR_PREIDXIND,2);
    regOp(INS_EOR,0x51,ADR_POSTIDXIND,2);

    // INC:
    regOp(INS_INC,0xE6,ADR_ZP,2);
    regOp(INS_INC,0xF6,ADR_ZPX,2);
    regOp(INS_INC,0xEE,ADR_ABS,3);
    regOp(INS_INC,0xFE,ADR_ABSX,3);

    // INX:
    regOp(INS_INX,0xE8,ADR_IMP,1);

    // INY:
    regOp(INS_INY,0xC8,ADR_IMP,1);

    // JMP:
    regOp(INS_JMP,0x4C,ADR_ABS,3);
    regOp(INS_JMP,0x6C,ADR_IND/*ABS*/,3);

    // JSR:
    regOp(INS_JSR,0x20,ADR_ABS,3);

    // LDA:
    regOp(INS_LDA,0xA9,ADR_IMM,2);
    regOp(INS_LDA,0xA5,ADR_ZP,2);
    regOp(INS_LDA,0xB5,ADR_ZPX,2);
    regOp(INS_LDA,0xAD,ADR_ABS,3);
    regOp(INS_LDA,0xBD,ADR_ABSX,3);
    regOp(INS_LDA,0xB9,ADR_ABSY,3);
    regOp(INS_LDA,0xA1,ADR_PREIDXIND,2);
    regOp(INS_LDA,0xB1,ADR_POSTIDXIND,2);


    // LDX:
    regOp(INS_LDX,0xA2,ADR_IMM,2);
    regOp(INS_LDX,0xA6,ADR_ZP,2);
    regOp(INS_LDX,0xB6,ADR_ZPY,2);
    regOp(INS_LDX,0xAE,ADR_ABS,3);
    regOp(INS_LDX,0xBE,ADR_ABSY,3);

    // LDY:
    regOp(INS_LDY,0xA0,ADR_IMM,2);
    regOp(INS_LDY,0xA4,ADR_ZP,2);
    regOp(INS_LDY,0xB4,ADR_ZPX,2);
    regOp(INS_LDY,0xAC,ADR_ABS,3);
    regOp(INS_LDY,0xBC,ADR_ABSX,3);

    // LSR:
    regOp(INS_LSRA,0x4A,ADR_ACC,1);
    regOp(INS_LSR,0x46,ADR_ZP,2);
    regOp(INS_LSR,0x56,ADR_ZPX,2);
    regOp(INS_LSR,0x4E,ADR_ABS,3);
    regOp(INS_LSR,0x5E,ADR_ABSX,3);

    // NOP:
    regOp(INS_NOP,0xEA,ADR_IMP,1);

    // ORA:
    regOp(INS_ORA,0x09,ADR_IMM,2);
    regOp(INS_ORA,0x05,ADR_ZP,2);
    regOp(INS_ORA,0x15,ADR_ZPX,2);
    regOp(INS_ORA,0x0D,ADR_ABS,3);
    regOp(INS_ORA,0x1D,ADR_ABSX,3);
    regOp(INS_ORA,0x19,ADR_ABSY,3);
    regOp(INS_ORA,0x01,ADR_PREIDXIND,2);
    regOp(INS_ORA,0x11,ADR_POSTIDXIND,2);

    // PHA:
    regOp(INS_PHA,0x48,ADR_IMP,1);

    // PHP:
    regOp(INS_PHP,0x08,ADR_IMP,1);

    // PLA:
    regOp(INS_PLA,0x68,ADR_IMP,1);

    // PLP:
    regOp(INS_PLP,0x28,ADR_IMP,1);

    // ROL:
    regOp(INS_ROLA,0x2A,ADR_ACC,1);
    regOp(INS_ROL,0x26,ADR_ZP,2);
    regOp(INS_ROL,0x36,ADR_ZPX,2);
    regOp(INS_ROL,0x2E,ADR_ABS,3);
    regOp(INS_ROL,0x3E,ADR_ABSX,3);

    // ROR:
    regOp(INS_RORA,0x6A,ADR_ACC,1);
    regOp(INS_ROR,0x66,ADR_ZP,2);
    regOp(INS_ROR,0x76,ADR_ZPX,2);
    regOp(INS_ROR,0x6E,ADR_ABS,3);
    regOp(INS_ROR,0x7E,ADR_ABSX,3);

    // RTI:
    regOp(INS_RTI,0x40,ADR_IMP,1);

    // RTS:
    regOp(INS_RTS,0x60,ADR_IMP,1);

    // SBC:
    regOp(INS_SBC,0xE9,ADR_IMM,2);
    regOp(INS_SBC,0xE5,ADR_ZP,2);
    regOp(INS_SBC,0xF5,ADR_ZPX,2);
    regOp(INS_SBC,0xED,ADR_ABS,3);
    regOp(INS_SBC,0xFD,ADR_ABSX,3);
    regOp(INS_SBC,0xF9,ADR_ABSY,3);
    regOp(INS_SBC,0xE1,ADR_PREIDXIND,2);
    regOp(INS_SBC,0xF1,ADR_POSTIDXIND,2);

    // SEC:
    regOp(INS_SEC,0x38,ADR_IMP,1);

    // SED:
    regOp(INS_SED,0xF8,ADR_IMP,1);

    // SEI:
    regOp(INS_SEI,0x78,ADR_IMP,1);

    // STA:
    regOp(INS_STA,0x85,ADR_ZP,2);
    regOp(INS_STA,0x95,ADR_ZPX,2);
    regOp(INS_STA,0x8D,ADR_ABS,3);
    regOp(INS_STA,0x9D,ADR_ABSX,3);
    regOp(INS_STA,0x99,ADR_ABSY,3);
    regOp(INS_STA,0x81,ADR_PREIDXIND,2);
    regOp(INS_STA,0x91,ADR_POSTIDXIND,2);

    // STX:
    regOp(INS_STX,0x86,ADR_ZP,2);
    regOp(INS_STX,0x96,ADR_ZPY,2);
    regOp(INS_STX,0x8E,ADR_ABS,3);

    // STY:
    regOp(INS_STY,0x84,ADR_ZP,2);
    regOp(INS_STY,0x94,ADR_ZPX,2);
    regOp(INS_STY,0x8C,ADR_ABS,3);

    // TAX:
    regOp(INS_TAX,0xAA,ADR_IMP,1);

    // TAY:
    regOp(INS_TAY,0xA8,ADR_IMP,1);

    // TSX:
    regOp(INS_TSX,0xBA,ADR_IMP,1);

    // TXA:
    regOp(INS_TXA,0x8A,ADR_IMP,1);

    // TXS:
    regOp(INS_TXS,0x9A,ADR_IMP,1);

    // TYA:
    regOp(INS_TYA,0x98,ADR_IMP,1);
}

class initialize
{
  public:
    initialize()
    {
        // const strings
        initStrings();

        // mark all entries as invalid
        memset(opdata,-1,sizeof(opdata));

        // fill table entries
        fillAllEntries();
        registerCommonOpcodes();
    }
};
initialize startup;

} // end namespace opcode
} // end namespace atari
} // end namespace cule


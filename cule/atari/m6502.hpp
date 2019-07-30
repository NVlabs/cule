#pragma once

#include <cule/config.hpp>
#include <cule/macros.hpp>

#include <cule/atari/common.hpp>
#include <cule/atari/debug.hpp>
#include <cule/atari/mmc.hpp>
#include <cule/atari/internals.hpp>
#include <cule/atari/interrupt.hpp>
#include <cule/atari/opcodes.hpp>
#include <cule/atari/stack.hpp>
#include <cule/atari/state.hpp>

#include <cule/atari/types/types.hpp>

namespace cule
{
namespace atari
{

// alias of registers with wrapping
typedef bit_field<_reg8_t, 8> reg_bit_field_t;
#define cule_regA fast_cast(s.A, reg_bit_field_t)
#define cule_regX fast_cast(s.X, reg_bit_field_t)
#define cule_regY fast_cast(s.Y, reg_bit_field_t)
#define cule_M fast_cast(s.value, operandb_t)
#define cule_SUM fast_cast(temp, alu_t)
#define cule_EA s.addr
#define NOTSAMEPAGE(_addr1, _addr2) (((_addr1) ^ (_addr2)) & 0xff00)

template<typename MMC_t,
         typename Stack_t,
         typename Interrupt_t>
struct m6502
{

static
CULE_ANNOTATION
uint8_t to_BCD(const uint8_t& t)
{
    return ((t >> 4) * 10) + (t & 0x0F);
}

static
CULE_ANNOTATION
uint8_t from_BCD(const uint8_t& t)
{
    return uint8_t(((t % 100) / 10) << 4) | uint8_t(t % 10);
}

template <class State_t, class T, int bits>
static
CULE_ANNOTATION
void setZ(State_t& s, const bit_field<T,bits>& result)
{
    s.sysFlags.template change<FLAG_ZERO>(result.zero());
}

template <class State_t, class T, int bits>
static
CULE_ANNOTATION
void setN(State_t& s, const bit_field<T,bits>& result)
{
    s.sysFlags.template change<FLAG_NEGATIVE>(result.negative());
}

template <class State_t, class T, int bits>
static
CULE_ANNOTATION
void setNZ(State_t& s, const bit_field<T,bits>& result)
{
    s.sysFlags.template change<FLAG_NEGATIVE>(result.negative());
    s.sysFlags.template change<FLAG_ZERO>(result.zero());
}

template <class State_t, class T, int bits>
static
CULE_ANNOTATION
void setNV(State_t& s, const bit_field<T,bits>& result)
{
    s.sysFlags.template copy<FLAG__NV, 6, 2>(result);
}

template <class State_t, class T, int bits>
static
CULE_ANNOTATION
void ASL(State_t& s, bit_field<T,bits>& operand)
{
    // Arithmetic Shift Left
    s.sysFlags.template change<FLAG_CARRY>(MSB(operand));
    operand.selfShl1();
    setNZ(s,operand);
}

template <class State_t, class T, int bits>
static
CULE_ANNOTATION
void LSR(State_t& s, bit_field<T,bits>& operand)
{
    // Logical Shift Right
    s.sysFlags.template change<FLAG_CARRY>(LSB(operand));
    operand.selfShr1();
    setZ(s,operand);
    s.sysFlags.clear(FLAG_SIGN);
}

template <class State_t, class T, int bits>
static
CULE_ANNOTATION
void ROL(State_t& s, bit_field<T,bits>& operand)
{
    // Rotate Left With Carry
    const bool newCarry=MSB(operand);
    operand.selfRcl(s.sysFlags[FLAG_CARRY]);
    s.sysFlags.template change<FLAG_CARRY>(newCarry);
    setNZ(s,operand);
}

template <class State_t, class T, int bits>
static
CULE_ANNOTATION
void ROR(State_t& s, bit_field<T,bits>& operand)
{
    // Rotate Right With Carry
    const bool newCarry=LSB(operand);
    operand.selfRcr(s.sysFlags[FLAG_CARRY]);
    s.sysFlags.template change<FLAG_CARRY>(newCarry);
    setNZ(s,operand);
}

template<typename State_t>
static
CULE_ANNOTATION
void ADC(State_t& s)
{
    int16_t temp = s.A + s.value + s.sysFlags[FLAG_CARRY];

    if (s.sysFlags[FLAG_BCD])
    {
        if(((temp & 0x0F) > 0x09) || ((s.A ^ s.value ^ temp) & 0x10))
        {
            temp += 0x06;
        }
        if(((temp & 0xF0) > 0x90) || (temp & 0x100))
        {
            temp += 0x60;
        }
    }

    s.sysFlags.template change<FLAG_OVERFLOW>(((s.A^temp)&0x80) && ((s.value^temp)&0x80));
    s.sysFlags.template change<FLAG_CARRY>((temp & 0x100) == 0x100);
    setNZ(s,cule_regA(temp & 0xFF));
}

template<typename State_t>
static
CULE_ANNOTATION
void SBC(State_t& s)
{
    if (s.sysFlags[FLAG_BCD])
    {
        int16_t temp = to_BCD(s.A) - to_BCD(s.value) - !s.sysFlags[FLAG_CARRY];
        temp = from_BCD(temp + ((temp<0) * 100));

        s.sysFlags.template change<FLAG_OVERFLOW>(((s.A^temp)&0x80) && ((s.value^temp)&0x80));
        s.sysFlags.template change<FLAG_CARRY>(s.A >= (s.value + !s.sysFlags[FLAG_CARRY]));

        setNZ(s,cule_regA(temp));
    }
    else
    {
        s.value = ~s.value;
        ADC(s);
        s.value = ~s.value;
    }
}

template<typename State_t>
static
CULE_ANNOTATION
uint8_t PS(State_t& s)
{
    return SELECT_FIELD(s.sysFlags.asBitField(), FIELD_SYS_PS);
}

template<typename State_t>
static
CULE_ANNOTATION
void reset(State_t& s)
{
    // reset general purpose registers
    s.A=0;
    s.X=0;
    s.Y=0;
    s.PC=0;

    // reset status register
    UPDATE_FIELD(s.sysFlags.asBitField(), FIELD_SYS_PS, 0x00);
    s.sysFlags.set(FLAG_BREAK);
    s.sysFlags.set(FLAG_RESERVED);
    s.sysFlags.clear(FLAG_CPU_HALT);
    s.sysFlags.clear(FLAG_CPU_LAST_READ);

    // reset stack pointer
    Stack_t::reset(s);

    // reset IRQ state
    Interrupt_t::clearAll(s);
    // this will set PC to the entry point
    Interrupt_t::request(s, FLAG_INT_RST);

    // reset cycles
    s.cpuCycles = 0;
}

template<typename State_t>
static
CULE_ANNOTATION
void readEffectiveAddress(State_t& s, const opcode::M6502_OPCODE& op, const bool forWriteOnly = false)
{
    invalidate(cule_EA);
    invalidate(cule_M);

    switch (op.addrmode)
    {
        case opcode::ADR_IMP: // Ignore. Address is implied in instruction.
        {
            break;
        }
        case opcode::ADR_ZP: // Zero Page mode. Use the address given after the opcode, but without high byte.
        {
            s.addr=MMC_t::fetchByteOperand(s,s.PC);
            if (!forWriteOnly) s.value=MMC_t::loadZPByte(s,s.addr);
            break;
        }
        case opcode::ADR_REL: // Relative mode.
        {
            s.addr=valueOf(MMC_t::fetchByteOperand(s,s.PC));
            if (s.addr[7])
            {
                // sign extension
                s.addr|=maddr_t(0xFF00);
            }
            s.addr+=s.PC;
            break;
        }
        case opcode::ADR_ABS: // Absolute mode. Use the two bytes following the opcode as an address.
        {
            s.addr=MMC_t::fetchWordOperand(s,s.PC);
            if (!forWriteOnly) s.value=MMC_t::read(s,s.addr);
            break;
        }
        case opcode::ADR_IMM: //Immediate mode. The value is given after the opcode.
        {
            s.addr=s.PC;
            if (!forWriteOnly) s.value=MMC_t::fetchByteOperand(s,s.PC);
            break;
        }
        case opcode::ADR_ZPX:
        {
            // Zero Page Indexed mode, X as index. Use the address given
            // after the opcode, then add the
            // X register to it to get the final address.
            s.addr=MMC_t::fetchByteOperand(s,s.PC).plus(s.X);
            if (!forWriteOnly) s.value=MMC_t::loadZPByte(s,s.addr);
            break;
        }
        case opcode::ADR_ZPY:
        {
            // Zero Page Indexed mode, Y as index. Use the address given
            // after the opcode, then add the
            // Y register to it to get the final address.
            s.addr=MMC_t::fetchByteOperand(s,s.PC).plus(s.Y);
            if (!forWriteOnly) s.value=MMC_t::loadZPByte(s,s.addr);
            break;
        }
        case opcode::ADR_ABSX:
        {
            // Absolute Indexed Mode, X as index. Same as zero page
            // indexed, but with the high byte.
            s.addr=MMC_t::fetchWordOperand(s,s.PC);
            if (NOTSAMEPAGE(s.addr, s.addr+s.X)) ++s.cpuCycles;
            s.addr+=s.X;
            if (!forWriteOnly) s.value=MMC_t::read(s,s.addr);
            break;
        }
        case opcode::ADR_ABSY:
        {
            // Absolute Indexed Mode, Y as index. Same as zero page
            // indexed, but with the high byte.
            s.addr=MMC_t::fetchWordOperand(s,s.PC);
            if (NOTSAMEPAGE(s.addr, s.addr+s.Y)) ++s.cpuCycles;
            s.addr+=s.Y;
            if (!forWriteOnly) s.value=MMC_t::read(s,s.addr);
            break;
        }
        case opcode::ADR_INDX:
        {
            s.addr=MMC_t::fetchByteOperand(s,s.PC).plus(s.X);
            s.addr=MMC_t::loadZPWord(s,s.addr);
            if (!forWriteOnly) s.value=MMC_t::read(s,s.addr);
            break;
        }
        case opcode::ADR_INDY:
        {
            s.addr=MMC_t::fetchByteOperand(s,s.PC);
            s.addr=MMC_t::loadZPWord(s,s.addr);
            if (NOTSAMEPAGE(s.addr, s.addr+s.Y)) ++s.cpuCycles;
            s.addr+=s.Y;
            if (!forWriteOnly) s.value=MMC_t::read(s,s.addr);
            break;
        }
        case opcode::ADR_IND:
        {
            // Indirect Absolute mode. Find the 16-bit address contained
            // at the given location.
            s.addr=MMC_t::fetchWordOperand(s,s.PC);
            maddr_t high = NOTSAMEPAGE(s.addr, s.addr + 1) ? maddr_t(s.addr & 0xFF00) : maddr_t(s.addr + 1);
            s.addr=makeWord(MMC_t::read(s,s.addr),MMC_t::read(s,high));
            if (!forWriteOnly) s.value=MMC_t::read(s,s.addr);
            break;
        }
        default:
        {
            /* FATAL_ERROR(INVALID_INSTRUCTION, INVALID_ADDRESS_MODE, "opcode", code, "instruction", op.inst, "adrmode", op.addrmode); */
            break;
        }
    }
}

template<typename State_t>
static
CULE_ANNOTATION
void execute(State_t& s, const opcode::M6502_OPCODE& op)
{
    switch (op.inst)
    {
        // arithmetic
        case opcode::INS_ADC: // Add with carry.
        {
            ADC(s);
            break;
        }
        case opcode::INS_SBC: // Subtract
        {
            SBC(s);
            break;
        }
        case opcode::INS_INC: // Increment memory by one
        {
            inc(cule_M);
            setNZ(s,cule_M);
            s.sysFlags.set(FLAG_CPU_WRITE_BACK);
            break;
        }
        case opcode::INS_DEC: // Decrement memory by one
        {
            dec(cule_M);
            setNZ(s,cule_M);
            s.sysFlags.set(FLAG_CPU_WRITE_BACK);
            break;
        }
        case opcode::INS_DEX: // Decrement index X by one
        {
            dec(cule_regX);
            setNZ(s,cule_regX);
            break;
        }
        case opcode::INS_DEY: // Decrement index Y by one
        {
            dec(cule_regY);
            setNZ(s,cule_regY);
            break;
        }
        case opcode::INS_INX: // Increment index X by one
        {
            inc(cule_regX);
            setNZ(s,cule_regX);
            break;
        }
        case opcode::INS_INY: // Increment index Y by one
        {
            inc(cule_regY);
            setNZ(s,cule_regY);
            break;
        }
        // bit manipulation
        case opcode::INS_AND: // AND memory with accumulator.
        {
            setNZ(s,cule_regA&=cule_M);
            break;
        }
        case opcode::INS_ASLA: // Shift left one bit
        {
            ASL(s,cule_regA);
            break;
        }
        case opcode::INS_ASL:
        {
            ASL(s,cule_M);
            s.sysFlags.set(FLAG_CPU_WRITE_BACK);
            break;
        }
        case opcode::INS_EOR: // XOR Memory with accumulator, store in accumulator
        {
            setNZ(s,cule_regA^=cule_M);
            break;
        }
        case opcode::INS_LSR: // Shift right one bit
        {
            LSR(s,cule_M);
            s.sysFlags.set(FLAG_CPU_WRITE_BACK);
            break;
        }
        case opcode::INS_LSRA:
        {
            LSR(s,cule_regA);
            break;
        }
        case opcode::INS_ORA: // OR memory with accumulator, store in accumulator.
        {
            setNZ(s,cule_regA|=cule_M);
            break;
        }
        case opcode::INS_ROL: // Rotate one bit left
        {
            ROL(s,cule_M);
            s.sysFlags.set(FLAG_CPU_WRITE_BACK);
            break;
        }
        case opcode::INS_ROLA:
        {
            ROL(s,cule_regA);
            break;
        }
        case opcode::INS_ROR: // Rotate one bit right
        {
            ROR(s,cule_M);
            s.sysFlags.set(FLAG_CPU_WRITE_BACK);
            break;
        }
        case opcode::INS_RORA:
        {
            ROR(s,cule_regA);
            break;
        }
        // branch
        case opcode::INS_JMP: // Jump to new location
        {
            s.PC=s.addr;
            break;
        }
        case opcode::INS_JSR: // Jump to new location, saving return address. Push return address on stack
        {
            dec(s.PC);
            Stack_t::pushPC(s);
            s.PC=s.addr;
            break;
        }
        case opcode::INS_RTS: // Return from subroutine. Pull PC from stack.
        {
            s.PC=Stack_t::popWord(s);
            inc(s.PC);
            break;
        }
        case opcode::INS_BCC: // Branch on carry clear
        {
            if (!s.sysFlags[FLAG_CARRY])
            {
                jBranch:
                s.cpuCycles += (NOTSAMEPAGE(s.PC, s.addr)) ? 2 : 1;
                s.PC=s.addr;
            }
            break;
        }
        case opcode::INS_BCS: // Branch on carry set
        {
            if (s.sysFlags[FLAG_CARRY]) goto jBranch;
            else break;
        }
        case opcode::INS_BEQ: // Branch on zero
        {
            if (s.sysFlags[FLAG_ZERO]) goto jBranch;
            else break;
        }
        case opcode::INS_BMI: // Branch on negative result
        {
            if (s.sysFlags[FLAG_SIGN]) goto jBranch;
            else break;
        }
        case opcode::INS_BNE: // Branch on not zero
        {
            if (!s.sysFlags[FLAG_ZERO]) goto jBranch;
            else break;
        }
        case opcode::INS_BPL: // Branch on positive result
        {
            if (!s.sysFlags[FLAG_SIGN]) goto jBranch;
            else break;
        }
        case opcode::INS_BVC: // Branch on overflow clear
        {
            if (!s.sysFlags[FLAG_OVERFLOW]) goto jBranch;
            else break;
        }
        case opcode::INS_BVS: // Branch on overflow set
        {
            if (s.sysFlags[FLAG_OVERFLOW]) goto jBranch;
            else break;
        }
        // interrupt
        case opcode::INS_BRK: // Break
        {
            inc(s.PC);
            Interrupt_t::request(s,FLAG_INT_BRK);
            Interrupt_t::poll(s);
            break;
        }
        case opcode::INS_RTI: // Return from interrupt. Pull status and PC from stack.
        {
            UPDATE_FIELD(s.sysFlags.asBitField(), FIELD_SYS_PS, Stack_t::popByte(s));
            s.sysFlags.set(FLAG_RESERVED);
            s.PC=Stack_t::popWord(s);
            break;
        }
        // set/clear flag
        case opcode::INS_CLC: // Clear carry flag
        {
            s.sysFlags.clear(FLAG_CARRY);
            break;
        }
        case opcode::INS_CLD: // Clear decimal flag
        {
            s.sysFlags.clear(FLAG_DECIMAL);
            break;
        }
        case opcode::INS_CLI: // Clear interrupt flag
        {
            s.sysFlags.clear(FLAG_INTERRUPT_OFF);
            break;
        }
        case opcode::INS_CLV: // Clear overflow flag
        {
            s.sysFlags.clear(FLAG_OVERFLOW);
            break;
        }
        case opcode::INS_SEC: // Set carry flag
        {
            s.sysFlags.set(FLAG_CARRY);
            break;
        }
        case opcode::INS_SED: // Set decimal flag
        {
            s.sysFlags.set(FLAG_DECIMAL);
            break;
        }
        case opcode::INS_SEI: // Set interrupt disable status
        {
            s.sysFlags.set(FLAG_INTERRUPT_OFF);
            break;
        }
        // compare
        case opcode::INS_BIT:
        {
            setNV(s,cule_M);
            setZ(s,cule_M&=s.A);
            break;
        }
        case opcode::INS_CMP: // Compare memory and accumulator
        case opcode::INS_CPX: // Compare memory and index X
        case opcode::INS_CPY: // Compare memory and index Y
        {
            _alutemp_t temp = 0;
            switch (op.inst)
            {
            case opcode::INS_CMP:
                temp=s.A;
                break;
            case opcode::INS_CPX:
                temp=s.X;
                break;
            case opcode::INS_CPY:
                temp=s.Y;
                break;
            default:
                break;
            }
            temp += (0x100 - s.value);
            s.sysFlags.template change<FLAG_CARRY>(cule_SUM.overflow());
            temp=(temp - 0x100) & 0xFF;
            setNZ(s,cule_SUM);
            break;
        }
        // load/store
        case opcode::INS_LDA: // Load accumulator with memory
        {
            setNZ(s,cule_M);
            s.A=s.value;
            break;
        }
        case opcode::INS_LDX: // Load index X with memory
        {
            setNZ(s,cule_M);
            s.X=s.value;
            break;
        }
        case opcode::INS_LDY: // Load index Y with memory
        {
            setNZ(s,cule_M);
            s.Y=s.value;
            break;
        }
        case opcode::INS_STA: // Store accumulator in memory
        {
            s.value = s.A;
            s.sysFlags.set(FLAG_CPU_WRITE_BACK);
            break;
        }
        case opcode::INS_STX: // Store index X in memory
        {
            s.value = s.X;
            s.sysFlags.set(FLAG_CPU_WRITE_BACK);
            break;
        }
        case opcode::INS_STY: // Store index Y in memory
        {
            s.value = s.Y;
            s.sysFlags.set(FLAG_CPU_WRITE_BACK);
            break;
        }
        // stack
        case opcode::INS_PHA: // Push accumulator on stack
        {
            Stack_t::pushReg(s,cule_regA);
            break;
        }
        case opcode::INS_PHP: // Push processor status on stack
        {
            reg_bit_field_t P(SELECT_FIELD(s.sysFlags.asBitField(), FIELD_SYS_PS));
            Stack_t::pushReg(s,P);
            break;
        }
        case opcode::INS_PLA: // Pull accumulator from stack
        {
            MMC_t::read(s, maddr_t(0x0100 + s.SP));
            s.A=Stack_t::popByte(s);
            setNZ(s,cule_regA);
            break;
        }
        case opcode::INS_PLP: // Pull processor status from stack
        {
            UPDATE_FIELD(s.sysFlags.asBitField(), FIELD_SYS_PS, Stack_t::popByte(s));
            s.sysFlags.set(FLAG_RESERVED);
            break;
        }
        // transfer
        case opcode::INS_TAX: // Transfer accumulator to index X
        {
            s.X=s.A;
            setNZ(s,cule_regX);
            break;
        }
        case opcode::INS_TAY: // Transfer accumulator to index Y
        {
            s.Y=s.A;
            setNZ(s,cule_regY);
            break;
        }
        case opcode::INS_TSX: // Transfer stack pointer to index X
        {
            s.X=valueOf(s.SP);
            setNZ(s,cule_regX);
            break;
        }
        case opcode::INS_TXA: // Transfer index X to accumulator
        {
            s.A=s.X;
            setNZ(s,cule_regA);
            break;
        }
        case opcode::INS_TXS: // Transfer index X to stack pointer
        {
            s.SP=s.X;
            break;
        }
        case opcode::INS_TYA: // Transfer index Y to accumulator
        {
            s.A=s.Y;
            setNZ(s,cule_regA);
            break;
        }
        // other
        case opcode::INS_NOP:
        {
            break; // No OPeration
        }
        // unofficial
        default:
        {
            s.sysFlags.set(FLAG_CPU_ERROR);
        }
    }
}

template<typename State_t>
static
CULE_ANNOTATION
void nextInstruction(State_t& s)
{
    // handle interrupt request
    Interrupt_t::poll(s);

    // step1: fetch instruction
    const maddr_t opaddr = s.PC;
    const opcode_t opcode = MMC_t::fetchOpcode(s,s.PC);

    // step2: decode
    const opcode::M6502_OPCODE op = opcode::decode(opcode);
    ERROR_UNLESS(opcode::usual(opcode::usualOp,opcode), INVALID_INSTRUCTION, INVALID_OPCODE, "opaddr", valueOf(opaddr), "opcode", opcode, "instruction", op.inst);
    s.cpuCycles += op.cycles;

    // step3: read effective address & operands
    readEffectiveAddress(s, op, (op.inst==opcode::INS_STA || op.inst==opcode::INS_STX || op.inst==opcode::INS_STY));
    CULE_ASSERT((valueOf(s.PC)-valueOf(opaddr)) == op.size, "Invalid operation (" << std::hex << op.inst << ") specified");

    // step4: execute
    s.sysFlags.clear(FLAG_CPU_WRITE_BACK);
    execute(s, op);

    if(s.sysFlags[FLAG_CPU_ERROR])
    {
        // execution failed
        FATAL_ERROR(INVALID_INSTRUCTION, INVALID_OPCODE, "opaddr", valueOf(opaddr), "opcode", opcode, "instruction", op.inst);
        return;
    }

    // debug::printDisassembly<MMC_t>(s, s.PC, opcode, s.X, s.Y, s.addr, s.value);
    // debug::printCPUState(s.PC, s.A, s.X, s.Y, PS(s), s.SP, s.cpuCycles);

    // step5: write back (when needed)
    if(s.sysFlags[FLAG_CPU_WRITE_BACK])
    {
        CULE_ASSERT(s.addr != 0xCCCC, "Invalid address");
        MMC_t::write(s, s.addr, s.value);
    }

    // end of instruction pipeline

    CULE_ASSERT(s.sysFlags[FLAG_RESERVED], "Invalid PC state (RESERVED NOT SET)");
}

// emulate at most n instructions within specified cycles
template<typename State_t>
static
CULE_ANNOTATION
void run(State_t& s)
{
    for(int16_t n = 0; n < 25000; n++)
    {
        nextInstruction(s);
        if(s.sysFlags[FLAG_CPU_HALT] || s.sysFlags[FLAG_CPU_ERROR])
        {
            s.sysFlags.clear(FLAG_CPU_HALT);
            break;
        }
    }
}

}; // end namespace m6502
} // end namespace atari
} // end namespace cule


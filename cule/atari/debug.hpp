#pragma once

#include <cule/config.hpp>
#include <cule/macros.hpp>

#include <cule/atari/opcodes.hpp>
#include <cule/atari/internals.hpp>

#include <stdarg.h>

#ifdef __CUDACC__
#define WARN(TYPE, SUBTYPE, ...)
#define WARN_IF(E, TYPE, SUBTYPE, ...)

#define ERROR(TYPE, SUBTYPE, ...)
#define ERROR_IF(E, TYPE, SUBTYPE, ...)
#define ERROR_UNLESS(E, TYPE, SUBTYPE, ...)

#define FATAL_ERROR(TYPE, SUBTYPE, ...)
#define FATAL_ERROR_IF(E, TYPE, SUBTYPE, ...)
#define FATAL_ERROR_UNLESS(E, TYPE, SUBTYPE, ...)
#else
#define WARN(TYPE, SUBTYPE, ...) cule::atari::debug::warn(TYPE, SUBTYPE, __FUNCTION__, __LINE__, __VA_ARGS__, 0)
#define WARN_IF(E, TYPE, SUBTYPE, ...) if (E) WARN(TYPE, SUBTYPE, __VA_ARGS__)

#define ERROR(TYPE, SUBTYPE, ...) cule::atari::debug::error(TYPE, SUBTYPE, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__, 0)
#define ERROR_IF(E, TYPE, SUBTYPE, ...) if (E) ERROR(TYPE, SUBTYPE, __VA_ARGS__)
#define ERROR_UNLESS(E, TYPE, SUBTYPE, ...) if (!(E)) ERROR(TYPE, SUBTYPE, __VA_ARGS__)

#define FATAL_ERROR(TYPE, SUBTYPE, ...) cule::atari::debug::fatalError(TYPE, SUBTYPE, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__, 0)
#define FATAL_ERROR_IF(E, TYPE, SUBTYPE, ...) if (E) FATAL_ERROR(TYPE, SUBTYPE, __VA_ARGS__)
#define FATAL_ERROR_UNLESS(E, TYPE, SUBTYPE, ...) if (!(E)) FATAL_ERROR(TYPE, SUBTYPE, __VA_ARGS__)
#endif

namespace cule
{
namespace atari
{

namespace debug
{

static FILE* foutput = stdout;

template<typename MMC_t, typename State>
CULE_ANNOTATION
void printDisassembly(State& s,
                      const maddr_t pc, const opcode_t opcode,
                      const _reg8_t rx, const _reg8_t ry,
                      const maddr_t addr, const operand_t operand)
{
    const cule::atari::opcode::M6502_OPCODE op = cule::atari::opcode::decode(opcode);

    switch (op.size)
    {
        case 1:
        {
            printf("%04X  %02X        %-4s    ", valueOf(pc), opcode, opcode::instName(op.inst));
            break;
        }
        case 2:
        {
            printf("%04X  %02X %02X     %-4s  ", valueOf(pc), opcode, MMC_t::read(s,maddr_t(pc + 1)), opcode::instName(op.inst));
            break;
        }
        case 3:
        {
            printf("%04X  %02X %02X %02X  %-4s", valueOf(pc), opcode, MMC_t::read(s,maddr_t(pc + 1)), MMC_t::read(s,maddr_t(pc + 2)), opcode::instName(op.inst));
            break;
        }
        default:
        {
            printf("Invalid opcode size : %d\n", int(op.size));
            assert(false);
        }
    }

    switch (op.addrmode)
    {
        case cule::atari::opcode::ADR_IMP:
        {
            printf("         ");
            break;
        }
        case cule::atari::opcode::ADR_ZP:
        case cule::atari::opcode::ADR_ZPX:
        case cule::atari::opcode::ADR_ZPY:
        {
            printf(" $%02X = %02X  ", valueOf(addr), operand);
            break;
        }
        case cule::atari::opcode::ADR_REL:
        {
            printf(" to $%04X  ", valueOf(addr));
            break;
        }
        case cule::atari::opcode::ADR_ABS:
        case cule::atari::opcode::ADR_ABSX:
        case cule::atari::opcode::ADR_ABSY:
        case cule::atari::opcode::ADR_INDX:
        case cule::atari::opcode::ADR_INDY:
        case cule::atari::opcode::ADR_IND:
        {
            printf(" $%04X = %02X  ", valueOf(addr), operand);
            break;
        }
        case cule::atari::opcode::ADR_IMM:
        {
            printf(" #$%02X      ", operand);
            break;
        }
        case cule::atari::opcode::ADR_INDABSX:
        case cule::atari::opcode::ADR_INDZP:
        case cule::atari::opcode::_ADR_MAX:
        case cule::atari::opcode::_ADR_INVALID:
        {
            printf("UNKNOWN ADDRESSING $%04X", valueOf(addr));
            break;
        }
    }
    printf("  ");
}

CULE_ANNOTATION
void printCPUState(const maddr_t pc, const _reg8_t ra, const _reg8_t rx, const _reg8_t ry, const _reg8_t rp, const _reg8_t rsp, const int cyc)
{
    printf("[A:%02X X:%02X Y:%02X P:%02X SP:%02X CYC:%3d] -> %04X\n", ra, rx, ry, rp, rsp, cyc, valueOf(pc));
}

void printPPUState(const long long frameNum, const int scanline, const bool vblank, const bool hit, const bool bgmsk, const bool sprmsk)
{
    // fprintf(foutput, "----- FR: %I64d SL: %3d VB:%s HIT:%s MSK:%c%c -----\n", frameNum, scanline, vblank?"True":"false", hit?"Yes":"no",
    fprintf(foutput, "----- FR: %lld SL: %3d VB:%s HIT:%s MSK:%c%c -----\n", frameNum, scanline, vblank?"True":"false", hit?"Yes":"no",
            bgmsk?'B':'_', sprmsk?'S':'_');
}

// a NULL at the end of argv is REQUIRED!
static void printToConsole(int type, const char * typestr, int stype, const char * stypestr, const char * file, const char * function_name, unsigned long line_number, va_list argv)
{
    printf("Type: %s (%d)\nSub Type: %s (%d)\nProc: %s:%ld\n", typestr, type, stypestr, stype, function_name, line_number);
    if (file != nullptr)
    {
        printf("File: %s\n", file);
    }

    // print custom parameters
    char* name=nullptr;
    while ((name=va_arg(argv, char*))!=nullptr)
    {
        int value=va_arg(argv, int);
        printf("<%s> = %Xh (%d)\n", name, value, value);
    }
}

static const char * errorTypeToString(EMUERROR type)
{
    switch (type)
    {
        CASE_ENUM_RETURN_STRING(INVALID_ROM);
        CASE_ENUM_RETURN_STRING(INVALID_MEMORY_ACCESS);
        CASE_ENUM_RETURN_STRING(INVALID_INSTRUCTION);
        CASE_ENUM_RETURN_STRING(ILLEGAL_OPERATION);

    default:
        return "UNKNOWN";
    }
}

static const char * errorSTypeToString(EMUERRORSUBTYPE stype)
{
    switch (stype)
    {
        CASE_ENUM_RETURN_STRING(INVALID_FILE_SIGNATURE);
        CASE_ENUM_RETURN_STRING(INVALID_ROM_CONFIG);
        CASE_ENUM_RETURN_STRING(UNEXPECTED_END_OFLAG_FILE);
        CASE_ENUM_RETURN_STRING(UNSUPPORTED_MAPPER_TYPE);

        CASE_ENUM_RETURN_STRING(MAPPER_FAILURE);
        CASE_ENUM_RETURN_STRING(ADDRESS_OUT_OFLAG_RANGE);
        CASE_ENUM_RETURN_STRING(ILLEGAL_ADDRESS_WARP);
        CASE_ENUM_RETURN_STRING(MEMORY_NOT_EXECUTABLE);
        CASE_ENUM_RETURN_STRING(MEMORY_CANT_BE_READ);
        CASE_ENUM_RETURN_STRING(MEMORY_CANT_BE_WRITTEN);
        CASE_ENUM_RETURN_STRING(MEMORY_CANT_BE_COPIED);

        CASE_ENUM_RETURN_STRING(INVALID_OPCODE);
        CASE_ENUM_RETURN_STRING(INVALID_ADDRESS_MODE);

        CASE_ENUM_RETURN_STRING(IRQ_ALREADY_PENDING);

    default:
        return "UNKNOWN";
    }
}

void fatalError(EMUERROR type, EMUERRORSUBTYPE stype, const char * file, const char * function_name, unsigned long line_number, ...)
{
    va_list args;
    va_start(args, line_number);
    printf("[X] Fatal error: \n");
    printToConsole(type, errorTypeToString(type), stype, errorSTypeToString(stype), file, function_name, line_number, args);
    va_end(args);
    fflush(foutput);
#ifndef NDEBUG
    assert(0);
#endif
    exit(type);
}

void error(EMUERROR type, EMUERRORSUBTYPE stype, const char * file, const char * function_name, unsigned long line_number, ...)
{
    va_list args;
    va_start(args, line_number);
    printf("[X] Error: \n");
    printToConsole(type, errorTypeToString(type), stype, errorSTypeToString(stype), file, function_name, line_number, args);
    va_end(args);
    fflush(foutput);
#ifndef NDEBUG
    assert(0);
#else
    // __debugbreak();
#endif
}

void warn(EMUERROR type, EMUERRORSUBTYPE stype, const char * function_name, unsigned long line_number, ...)
{
    va_list args;
    va_start(args, line_number);
    printf("[!] Warning: \n");
    printToConsole(type, errorTypeToString(type), stype, errorSTypeToString(stype), NULL, function_name, line_number, args);
    va_end(args);
}

void setOutputFile(FILE *fp)
{
    foutput = fp;
}

} // end namespace debug
} // end namespace atari
} // end namespace cule


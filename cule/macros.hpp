#pragma once

#include <cule/config.hpp>

namespace cule
{

// e.g. SIZE_IN_BITS(int) is 32 and SIZE_IN_BITS(bool) is 1
#define SIZE_IN_BITS(TP) (__BITSOFLAG_CLASS<TP>::NUM_BITS)

// e.g. BIT_MASK(3)=7
// bits must be positive
#define BIT_MASK(TP, bits) (1|((((TP)1<<((bits)-1))-1)<<1))

// e.g. TYPE_MAX(int)=0xFFFFFFFF
#define TYPE_MAX(TP) BIT_MASK(TP,SIZE_IN_BITS(TP))

// bit manipulation
#define LOW_BIT(x) ((x)&(-(x)))
#define RTRIM(x) ((x)/LOW_BIT(x))

#define SINGLE_BIT(x) (((x)&((x)-1))==0)

#define SELECT_FIELD(x, f) (((x)&(f))/LOW_BIT(f))
#define UPDATE_FIELD(x, f, y) x=((x)&(~f))|(((y)*LOW_BIT(f))&(f))
#define INC_FIELD(x, f) x=((x)&(~(f))) | ( (((x)&(f)) + LOW_BIT(f)) & (f) );

#define CASE_ENUM_RETURN_STRING(ENUM) case ENUM: return #ENUM

#define fast_cast(VAR,TP) ((TP&)*(TP*)(&(VAR)))
#define fast_constcast(VAR,TP) ((const TP&)*(const TP*)(&(VAR)))

// type casting
template <typename destType, typename srcType>
destType& safe_cast(srcType& source)
{
    return *(destType*)(&source);
}

template <typename T>
CULE_ANNOTATION
T min(const T& x,const T& y)
{
    return x<y?x:y;
}

template <typename T>
CULE_ANNOTATION
T max(const T& x,const T& y)
{
    return x>y?x:y;
}

// Type info
template <typename T>
class __BITSOFLAG_CLASS
{
public:
    __BITSOFLAG_CLASS();
    enum :int {
        NUM_BITS=sizeof(T)<<3
    };
};

template <>
class __BITSOFLAG_CLASS<bool>
{
public:
    __BITSOFLAG_CLASS();
    enum :int {
        NUM_BITS=1
    };
};

} // end namespace cule


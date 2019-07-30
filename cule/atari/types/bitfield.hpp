#pragma once

#include <cule/config.hpp>
#include <cule/macros.hpp>

#include <cule/atari/types/valueobj.hpp>

#include <iostream>

namespace cule
{
namespace atari
{

#if (defined(__CUDA_ARCH__) || (__CUDACC_VER_MAJOR__ >= 9) || !defined(__CUDACC__))
#define FLIP_CHECK 1
#else
#define FLIP_CHECK 0
#endif

template <typename T, int bits>
// integral type of known, fixed bit-width
class bit_field: public value_object<T>
{
private:

    using super_t = value_object<T>;

public:
    enum :T
    {
        MASK=BIT_MASK(T,bits)
    };

protected:

    CULE_ANNOTATION
    static  bit_field _unchecked_wrapper(const T data)
    {
        bit_field ret;
        ret._value = data;
        return ret;
    }

public:

    // ctors
    CULE_ANNOTATION
    bit_field()
      : super_t()
    {};

    CULE_ANNOTATION
    bit_field(const bit_field& other)
      : super_t()
    {
        super_t::_value = other._value;   // copy ctor (w/o checking)
    }

    CULE_ANNOTATION
    bit_field& operator=(const bit_field& other)
    {
        super_t::_value = other._value;    // copy assignment ctor (w/o checking)
        return *this;
    }

    template <typename ET>
    CULE_ANNOTATION
    bit_field(const flag_set<T, ET, bits>& fs)
    {
        super_t::_value = valueOf(fs);
    }

    // restricted checked auto boxing
    CULE_ANNOTATION
    explicit bit_field(const T value)
    {
        super_t::_value = value;
    }

    // transparent value getter (auto unboxing)
    CULE_ANNOTATION
    operator T() const
    {
        return super_t::_value;
    }

    // checked setter
    CULE_ANNOTATION
    bit_field& operator = (const T value)
    {
        super_t::_value = value;
        return *this;
    }

    // unchecked setter for values within a not-bigger range
    template <class T2, int bits2>
    CULE_ANNOTATION
    bit_field& operator = (const bit_field<T2, bits2>& other)
    {
        super_t::_value = valueOf(other);
        return *this;
    }

    // auto-wrap value setter (no warning when value out of range)
    CULE_ANNOTATION
    bit_field& operator () (const T data)
    {
        #if FLIP_CHECK
        super_t::_value = data&(MASK);
        #else
        super_t::_value = data&(~MASK);
        #endif
        return *this;
    }

    // flag set converter (need to specify enum type)
    template <typename ET>
    CULE_ANNOTATION
    flag_set<T, ET, bits>& asFlagSet()
    {
        return *(flag_set<T,ET,bits>*)this;
    }

    // out of range checker
    CULE_ANNOTATION
    bool error() const
    {
        #if FLIP_CHECK
        return super_t::_value&(MASK);
        #else
        return super_t::_value&(~MASK);
        #endif
    }

    CULE_ANNOTATION
    void check() const
    {
    }

    // query
    CULE_ANNOTATION
    friend bool MSB(const bit_field& bf)
    {
        return 1&(bf._value>>(bits-1));
    }

    CULE_ANNOTATION
    friend bool LSB(const bit_field& bf)
    {
        return 1&bf._value;
    }

    CULE_ANNOTATION
    bool bitAt(const int n) const
    {
        return 0!=(super_t::_value&(((T)1)<<n));
    }

    CULE_ANNOTATION
    bool operator [](const int n) const
    {
        return bitAt(n);
    }

    CULE_ANNOTATION
    bool negative() const
    {
        return MSB(*this);
    }

    CULE_ANNOTATION
    bool zero() const
    {
        return !super_t::_value;
    }

    CULE_ANNOTATION
    bool belowMax() const
    {
        #if FLIP_CHECK
        return 0!=((~super_t::_value)&(MASK));
        #else
        return 0!=((~super_t::_value)&(~MASK));
        #endif
    }

    CULE_ANNOTATION
    bool reachMax() const
    {
        #if FLIP_CHECK
        return super_t::_value==T(MASK);
        #else
        return super_t::_value==T(~MASK);
        #endif
    }

    CULE_ANNOTATION
    bool overflow() const
    {
        return 0!=(super_t::_value>>bits);
    }

    CULE_ANNOTATION
    T lowbit() const
    {
        return LOW_BIT(super_t::_value);
    }

    CULE_ANNOTATION
    bit_field plus(const T delta) const
    {
        #if FLIP_CHECK
        return _unchecked_wrapper((super_t::_value+delta)&(MASK));
        #else
        return _unchecked_wrapper((super_t::_value+delta)&(~MASK));
        #endif
    }

    CULE_ANNOTATION
    bit_field minus(const T delta) const
    {
        #if FLIP_CHECK
        return _unchecked_wrapper((super_t::_value-delta)&(MASK));
        #else
        return _unchecked_wrapper((super_t::_value-delta)&(~MASK));
        #endif
    }

    // edit
    CULE_ANNOTATION
    bit_field& operator += (const T delta)
    {
        #if FLIP_CHECK
        super_t::_value=(super_t::_value+delta)&(MASK);
        #else
        super_t::_value=(super_t::_value+delta)&(~MASK);
        #endif
        return *this;
    }

    CULE_ANNOTATION
    bit_field& operator -= (const T delta)
    {
        #if FLIP_CHECK
        super_t::_value=(super_t::_value-delta)&(MASK);
        #else
        super_t::_value=(super_t::_value-delta)&(~MASK);
        #endif
        return *this;
    }

    CULE_ANNOTATION
    bit_field& operator &= (const T other)
    {
        super_t::_value&=other;
        return *this;
    }

    CULE_ANNOTATION
    bit_field& operator |= (const bit_field& other)
    {
        super_t::_value|=other._value;
        return *this;
    }

    CULE_ANNOTATION
    bit_field& operator ^= (const bit_field& other)
    {
        super_t::_value^=other._value;
        return *this;
    }

    CULE_ANNOTATION
    friend T inc(bit_field& bf)
    {
        ++bf._value;
        #if FLIP_CHECK
        return bf._value&=(MASK);
        #else
        return bf._value&=(~MASK);
        #endif
    }

    CULE_ANNOTATION
    friend T dec(bit_field& bf)
    {
        --bf._value;
        #if FLIP_CHECK
        return bf._value&=(MASK);
        #else
        return bf._value&=(~MASK);
        #endif
    }

    CULE_ANNOTATION
    friend void invalidate(bit_field& bf)
    {
        bf(T(0xCCCCCCCC));
    }

    CULE_ANNOTATION
    void selfRTrim()
    {
        super_t::_value=RTRIM(super_t::_value);
    }

    CULE_ANNOTATION
    void selfDropLowbit()
    {
        super_t::_value&=(super_t::_value-1);
    }

    CULE_ANNOTATION
    void selfNOT()
    {
        #if FLIP_CHECK
        super_t::_value^=(MASK); // faster
        #else
        super_t::_value^=(~MASK); // faster
        #endif
    }

    CULE_ANNOTATION
    void selfSetMax()
    {
        #if FLIP_CHECK
        super_t::_value=T(MASK);
        #else
        super_t::_value=T(~MASK);
        #endif
    }

    CULE_ANNOTATION
    void selfNEG()
    {
        #if FLIP_CHECK
        super_t::_value=(MASK)&(-super_t::_value);
        #else
        super_t::_value=(~MASK)&(-super_t::_value);
        #endif
    }

    // ShiftLeft
    CULE_ANNOTATION
    void selfShl(const int n)
    {
        #if FLIP_CHECK
        super_t::_value=(super_t::_value<<n)&(MASK);
        #else
        super_t::_value=(super_t::_value<<n)&(~MASK);
        #endif
    }

    CULE_ANNOTATION
    void selfShl1()
    {
        #if FLIP_CHECK
        super_t::_value=(super_t::_value<<1)&(MASK);
        #else
        super_t::_value=(super_t::_value<<1)&(~MASK);
        #endif
    }

    // ShiftRight
    CULE_ANNOTATION
    void selfShr(const int n)
    {
        super_t::_value>>=n;
    }

    CULE_ANNOTATION
    void selfShr1()
    {
        super_t::_value>>=1;
    }

    // RotateLeftWithCarry
    CULE_ANNOTATION
    void selfRcl(const bool carry)
    {
        #if FLIP_CHECK
        super_t::_value=((super_t::_value<<1)&(MASK))|(carry?1:0);
        #else
        super_t::_value=((super_t::_value<<1)&(~MASK))|(carry?1:0);
        #endif
    }

    // RotateLeft
    CULE_ANNOTATION
    void selfRol()
    {
        selfRcl(MSB(*this));
    }

    // RotateRightWithCarry
    CULE_ANNOTATION
    void selfRcr(const bool carry)
    {
        super_t::_value=(super_t::_value>>1)|(carry?(1<<(bits-1)):0);
    }

    // RotateRight
    CULE_ANNOTATION
    void selfRor()
    {
        selfRcr(LSB(*this));
    }
}; // end class bit_field

} // end namespace atari
} // end namespace cule


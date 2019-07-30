#pragma once

#include <cule/config.hpp>

#include <cule/atari/types/valueobj.hpp>

namespace cule
{
namespace atari
{

template <typename T, typename ET, int bits=SIZE_IN_BITS(T)>
//  integral type that represents a set of flags
class flag_set: public value_object<T>
{
private:

    using super_t = value_object<T>;

public:
    enum :T
    {
        MASK=BIT_MASK(T,bits)
    };

    // default ctor
    CULE_ANNOTATION
    flag_set():value_object<T>(0) {} // value initialized to zero

    // bit field converter
    CULE_ANNOTATION
    bit_field<T,bits>& asBitField()
    {
        return *(bit_field<T,bits>*)this;
    }

    // auto conversion is also fine
#ifndef DISABLE_FLAGSET_AUTO_UNBOXING
    CULE_ANNOTATION
    operator bit_field<T,bits>&() const {
        return *(bit_field<T,bits>*)this;
    }
#endif

    // safe assignment
    CULE_ANNOTATION
    flag_set& operator =(const bit_field<T,bits>& rhs)
    {
        super_t::_value=valueOf(rhs);
        return *this;
    }

    // safe auto boxing
    CULE_ANNOTATION
    flag_set(const bit_field<T,bits>& rhs) : value_object<T>(0)
    {
        super_t::_value=valueOf(rhs);
    }

    CULE_ANNOTATION
    bool any() const
    {
        return super_t::_value!=0;
    }

    CULE_ANNOTATION
    bool test(const ET e) const
    {
        return (super_t::_value&(T)e)==(T)e;
    }

    CULE_ANNOTATION
    bool operator [](const ET e) const
    {
        return test(e);
    }

    CULE_ANNOTATION
    void flip(const ET e)
    {
        super_t::_value^=(T)e;
    }

    CULE_ANNOTATION
    void operator ^=(const ET e)
    {
        return flip(e);
    }

    CULE_ANNOTATION
    void set(const ET e)
    {
        super_t::_value|=(T)e;
    }

    CULE_ANNOTATION
    void operator |=(const ET e)
    {
        set(e);
    }

    CULE_ANNOTATION
    void clear(const ET e)
    {
        super_t::_value&=(~(T)e);
    }

    CULE_ANNOTATION
    void operator -=(const ET e)
    {
        clear(e);
    }

    CULE_ANNOTATION
    void change(const ET e, const bool enabled)
    {
        super_t::_value=(super_t::_value&(~T(e)))|((-int(enabled))&T(e));
    }

    template <ET e>
    CULE_ANNOTATION
    void change(const int enabled)
    {
        super_t::_value=(super_t::_value&(~T(e)))|((-enabled)&T(e));
    }

    CULE_ANNOTATION
    void setAll()
    {
        super_t::_value=MASK;
    }

    CULE_ANNOTATION
    void clearAll()
    {
        super_t::_value=0;
    }

    // advanced bit manipulation
    CULE_ANNOTATION
    T mask(const ET e) const
    {
        return super_t::_value&((T)e);
    }

    CULE_ANNOTATION
    T select(const ET e) const
    {
        return SELECT_FIELD(super_t::_value, (T)e);
    }

    CULE_ANNOTATION
    T operator ()(const ET e) const
    {
        return SELECT_FIELD(super_t::_value, (T)e);
    }

    CULE_ANNOTATION
    void update(const ET e, const T newValue)
    {
        UPDATE_FIELD(super_t::_value, (T)e, newValue);
    }

    template <ET e>
    CULE_ANNOTATION
    void update(const T newValue)
    {
        UPDATE_FIELD(super_t::_value, (T)e, newValue);
    }

    CULE_ANNOTATION
    T inc(const ET e)
    {
        INC_FIELD(super_t::_value, (T)e);
        return SELECT_FIELD(super_t::_value, (T)e);;
    }

    template <ET e>
    CULE_ANNOTATION
    T inc()
    {
        INC_FIELD(super_t::_value, (T)e);
        return SELECT_FIELD(super_t::_value, (T)e);;
    }

    // set the specified field to the lower `length` bits of (src>>shift)
    CULE_ANNOTATION
    void copy(const ET e, const T src, const int shift, const int length=1)
    {
        update(e, (src>>shift)&BIT_MASK(T, length));
    }

    template <ET e, int shift, int length>
    CULE_ANNOTATION
    void copy(const T src)
    {
        const T field=(T)e;
        const T newValue = (src>>shift)&BIT_MASK(T, length);
        UPDATE_FIELD(super_t::_value, field, newValue);
    }

}; // end class flag_set

} // end namespace atari
} // end namespace cule


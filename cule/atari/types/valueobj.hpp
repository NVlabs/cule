#pragma once

#include <cule/config.hpp>

namespace cule
{
namespace atari
{

template <class VT, class DT = VT>
class value_object
{
public:
    typedef DT DataTp;
    typedef VT ValueTp;

    CULE_ANNOTATION
    value_object() {}

    CULE_ANNOTATION
    value_object(const VT& value): _value(value) {}

    CULE_ANNOTATION
    friend const VT& valueOf(const value_object& vo)
    {
        return vo._value;
    }

protected:
    DT _value;
};

template <class VT, class DT = VT>
class transparent_value_object : public value_object<VT, DT>
{
private:
    using super_t = value_object<VT, DT>;

public:
    CULE_ANNOTATION
    transparent_value_object() {}

    CULE_ANNOTATION
    transparent_value_object(const VT& value) {}

    // transparent value getter
    CULE_ANNOTATION
    operator const VT&() const
    {
        return super_t::_value;
    }

    // transparent value setter
    CULE_ANNOTATION
    transparent_value_object& operator = (const value_object<VT,DT>& other)
    {
        super_t::_value = other._value;
    }
};

} // end namespace atari
} // end namespace cule


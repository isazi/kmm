#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "fmt/ostream.h"

#include "kmm/utils/key_value_pair.hpp"

namespace kmm {

enum struct ScalarKind : uint8_t {
    Invalid = 0,
    Int8,
    Int16,
    Int32,
    Int64,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Float16,
    Float32,
    Float64,
    BFloat16,
    Complex16,
    Complex32,
    Complex64,
    KeyAndInt64,
    KeyAndFloat64,
};

template<typename T>
struct DataTypeOf;

struct DataType {
    template<typename T>
    static DataType of() {
        return DataTypeOf<T>::value;
    }

    DataType(ScalarKind ty = ScalarKind::Invalid) : m_kind(ty) {}

    ScalarKind get() const {
        return m_kind;
    }

    size_t alignment() const;
    size_t size_in_bytes() const;
    const char* c_name() const;
    const char* name() const noexcept;

    friend bool operator==(const DataType& a, const DataType& b) {
        return a.get() == b.get();
    }

    friend bool operator!=(const DataType& a, const DataType& b) {
        return !(a == b);
    }

  private:
    ScalarKind m_kind;
};

#define KMM_DEFINE_SCALAR_ALIAS(S, T) \
    template<>                        \
    struct DataTypeOf<T>: std::integral_constant<ScalarKind, ScalarKind::S> {};

KMM_DEFINE_SCALAR_ALIAS(Int8, int8_t)
KMM_DEFINE_SCALAR_ALIAS(Int16, int16_t)
KMM_DEFINE_SCALAR_ALIAS(Int32, int32_t)
KMM_DEFINE_SCALAR_ALIAS(Int64, int64_t)
KMM_DEFINE_SCALAR_ALIAS(Uint8, uint8_t)
KMM_DEFINE_SCALAR_ALIAS(Uint16, uint16_t)
KMM_DEFINE_SCALAR_ALIAS(Uint32, uint32_t)
KMM_DEFINE_SCALAR_ALIAS(Uint64, uint64_t)
KMM_DEFINE_SCALAR_ALIAS(Float32, float)
KMM_DEFINE_SCALAR_ALIAS(Float64, double)
KMM_DEFINE_SCALAR_ALIAS(Complex32, ::std::complex<float>)
KMM_DEFINE_SCALAR_ALIAS(Complex64, ::std::complex<double>)
KMM_DEFINE_SCALAR_ALIAS(KeyAndInt64, KeyValue<int64_t>)
KMM_DEFINE_SCALAR_ALIAS(KeyAndFloat64, KeyValue<double>)

std::ostream& operator<<(std::ostream& f, ScalarKind p);
std::ostream& operator<<(std::ostream& f, DataType p);

}  // namespace kmm

template<>
struct fmt::formatter<kmm::ScalarKind>: fmt::ostream_formatter {};

template<>
struct fmt::formatter<kmm::DataType>: fmt::ostream_formatter {};
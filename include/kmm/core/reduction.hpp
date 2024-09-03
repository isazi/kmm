#pragma once

#include <cstdef>
#include <cstdint>

namespace kmm {

enum struct ReductionOp { Sum, Product, Min, Max, BitAnd, BitOr };

enum struct DataType {
    Int8,
    Int16,
    Int32,
    Int64,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Float32,
    Float64,
    Complex32,
    Complex64
};

namespace detail {
template<typename T>
struct DataTypeOf {};

#define KMM_DEFINE_DTYPE_OF(T, V)     \
    template<>                        \
    struct DataTypeOf<T> {            \
        DataType value = DataType::V; \
    };

KMM_DEFINE_DTYPE_OF(int8_t, Int8);
KMM_DEFINE_DTYPE_OF(int16_t, Int16);
KMM_DEFINE_DTYPE_OF(int32_t, Int32);
KMM_DEFINE_DTYPE_OF(int64_t, Int64);
KMM_DEFINE_DTYPE_OF(uint8_t, Uint8);
KMM_DEFINE_DTYPE_OF(uint16_t, Uint16);
KMM_DEFINE_DTYPE_OF(uint32_t, Uint32);
KMM_DEFINE_DTYPE_OF(uint64_t, Uint64);
KMM_DEFINE_DTYPE_OF(float, Float32);
KMM_DEFINE_DTYPE_OF(double, Float64);
}  // namespace detail

template<typename T>
DataType data_type_of(const T& value = {}) {
    return detail::DataTypeOf<T>::value;
}

struct Reduction {
    ReductionOp op;
    DataType data_type;
};

}  // namespace kmm
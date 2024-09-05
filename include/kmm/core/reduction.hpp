#pragma once

#include <cstddef>
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
    Float16,
    Float32,
    Float64,
    Complex16,
    Complex32,
    Complex64
};

struct DataTypeInfo {
    DataType data_type;
    const char* type_name;
    const char* c_name;
    size_t size_in_bytes;
    size_t align;
};

struct Reduction {
    ReductionOp op;
    DataType data_type;
};

}  // namespace kmm
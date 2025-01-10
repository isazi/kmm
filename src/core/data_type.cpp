#include <complex>

#include "fmt/format.h"

#include "kmm/core/data_type.hpp"

namespace kmm {

struct DataTypeInfo {
    const std::type_info& type_id;
    size_t size_in_bytes;
    size_t alignment;
    const char* c_name;
};

DataTypeInfo get_info(DataType dtype) {
#define DTYPE_CASE(T) {typeid(T), sizeof(T), alignof(T), #T};

    switch (dtype.get()) {
        case ScalarKind::Int8:
            return DTYPE_CASE(int8_t);
        case ScalarKind::Int16:
            return DTYPE_CASE(int16_t);
        case ScalarKind::Int32:
            return DTYPE_CASE(int32_t);
        case ScalarKind::Int64:
            return DTYPE_CASE(int64_t);
        case ScalarKind::Uint8:
            return DTYPE_CASE(uint8_t);
        case ScalarKind::Uint16:
            return DTYPE_CASE(uint16_t);
        case ScalarKind::Uint32:
            return DTYPE_CASE(uint32_t);
        case ScalarKind::Uint64:
            return DTYPE_CASE(uint64_t);
        case ScalarKind::Float32:
            return DTYPE_CASE(float);
        case ScalarKind::Float64:
            return DTYPE_CASE(double);
        case ScalarKind::Complex32:
            return DTYPE_CASE(::std::complex<float>);
        case ScalarKind::Complex64:
            return DTYPE_CASE(::std::complex<double>);
        case ScalarKind::KeyAndInt64:
            return DTYPE_CASE(KeyValue<int64_t>);
        case ScalarKind::KeyAndFloat64:
            return DTYPE_CASE(KeyValue<double>);
        default:
            throw std::runtime_error(fmt::format("invalid data type: {}", dtype));
    }
}

size_t DataType::alignment() const {
    return get_info(*this).alignment;
}

size_t DataType::size_in_bytes() const {
    return get_info(*this).size_in_bytes;
}

const char* DataType::c_name() const {
    return get_info(*this).c_name;
}

const char* DataType::name() const noexcept {
    switch (m_kind) {
        case ScalarKind::Int8:
            return "Int8";
        case ScalarKind::Int16:
            return "Int16";
        case ScalarKind::Int32:
            return "Int32";
        case ScalarKind::Int64:
            return "Int64";
        case ScalarKind::Uint8:
            return "Uint8";
        case ScalarKind::Uint16:
            return "Uint16";
        case ScalarKind::Uint32:
            return "Uint32";
        case ScalarKind::Uint64:
            return "Uint64";
        case ScalarKind::Float16:
            return "Float16";
        case ScalarKind::Float32:
            return "Float32";
        case ScalarKind::Float64:
            return "Float64";
        case ScalarKind::BFloat16:
            return "BFloat16";
        case ScalarKind::Complex16:
            return "Float64";
        case ScalarKind::Complex32:
            return "Complex32";
        case ScalarKind::Complex64:
            return "Complex64";
        case ScalarKind::KeyAndInt64:
            return "KeyAndInt64";
        case ScalarKind::KeyAndFloat64:
            return "KeyAndFloat64";
        default:
            return "(unknown type)";
    }
}

std::ostream& operator<<(std::ostream& f, ScalarKind p) {
    return f << DataType(p);
}

std::ostream& operator<<(std::ostream& f, DataType p) {
    return f << p.name();
}

}  // namespace kmm
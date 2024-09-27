#include "kmm/core/reduction.hpp"

namespace kmm {

template<typename T>
std::vector<uint8_t> into_vector(const T& value) {
    uint8_t buffer[sizeof(T)];
    ::memcpy(buffer, &value, sizeof(T));
    return {buffer, buffer + sizeof(T)};
}

template<typename T>
std::vector<uint8_t> identity_value_for_type(ReductionOp op) {
    switch (op) {
        case ReductionOp::Sum:
        case ReductionOp::BitOr:
            return into_vector(T {0});
        case ReductionOp::Product:
        case ReductionOp::BitAnd:
            return into_vector(T {1});
        case ReductionOp::Min:
            return into_vector(std::numeric_limits<T>::max());
        case ReductionOp::Max:
            return into_vector(std::numeric_limits<T>::min());
        default:
            throw std::runtime_error("invalid reduction operation");
    }
}

std::vector<uint8_t> reduction_identity_value(DataType dtype, ReductionOp op) {
    switch (dtype.get()) {
        case ScalarKind::Int8:
            return identity_value_for_type<int8_t>(op);
        case ScalarKind::Int16:
            return identity_value_for_type<int16_t>(op);
        case ScalarKind::Int32:
            return identity_value_for_type<int32_t>(op);
        case ScalarKind::Int64:
            return identity_value_for_type<int64_t>(op);
        case ScalarKind::Uint8:
            return identity_value_for_type<uint8_t>(op);
        case ScalarKind::Uint16:
            return identity_value_for_type<uint16_t>(op);
        case ScalarKind::Uint32:
            return identity_value_for_type<uint32_t>(op);
        case ScalarKind::Uint64:
            return identity_value_for_type<uint64_t>(op);
        case ScalarKind::Float32:
            return identity_value_for_type<float>(op);
        case ScalarKind::Float64:
            return identity_value_for_type<double>(op);
        case ScalarKind::Complex32:
            return identity_value_for_type<std::complex<float>>(op);
        case ScalarKind::Complex64:
            return identity_value_for_type<std::complex<double>>(op);
        case ScalarKind::KeyAndInt64:
            return identity_value_for_type<KeyValue<int64_t>>(op);
        case ScalarKind::KeyAndFloat64:
            return identity_value_for_type<KeyValue<double>>(op);
        default:
            throw std::runtime_error(fmt::format("invalid reduction {} for type {}", op, dtype));
    }
}

std::ostream& operator<<(std::ostream& f, ReductionOp p) {
    switch (p) {
        case ReductionOp::Sum:
            return f << "Sum";
        case ReductionOp::Product:
            return f << "Product";
        case ReductionOp::Min:
            return f << "Min";
        case ReductionOp::Max:
            return f << "Max";
        case ReductionOp::BitAnd:
            return f << "BitAnd";
        case ReductionOp::BitOr:
            return f << "BitOr";
        default:
            return f << "(unknown operation)";
    }
}

std::ostream& operator<<(std::ostream& f, Reduction p) {
    return f << "Reduction(" << p.operation << ", " << p.data_type << ")";
}

}  // namespace kmm
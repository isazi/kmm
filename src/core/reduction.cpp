#include "../memops/host_reducers.hpp"

#include "kmm/core/reduction.hpp"

namespace kmm {

[[noreturn]] void throw_invalid_reduction_exception(DataType dtype, ReductionOp op) {
    throw std::runtime_error(fmt::format("invalid reduction {} for type {}", op, dtype));
}

template<typename T>
std::vector<uint8_t> into_vector(const T& value) {
    uint8_t buffer[sizeof(T)];
    ::memcpy(buffer, &value, sizeof(T));
    return {buffer, buffer + sizeof(T)};
}

template<typename T, ReductionOp Op>
std::vector<uint8_t> identity_value_for_type_and_op() {
    if constexpr (ReductionFunctorSupported<T, Op>()) {
        return into_vector(ReductionFunctor<T, Op>::identity());
    } else {
        throw_invalid_reduction_exception(DataType::of<T>(), Op);
    }
}

template<typename T>
std::vector<uint8_t> identity_value_for_type(ReductionOp op) {
    switch (op) {
        case ReductionOp::Sum:
            return identity_value_for_type_and_op<T, ReductionOp::Sum>();
        case ReductionOp::Product:
            return identity_value_for_type_and_op<T, ReductionOp::Product>();
        case ReductionOp::Min:
            return identity_value_for_type_and_op<T, ReductionOp::Min>();
        case ReductionOp::Max:
            return identity_value_for_type_and_op<T, ReductionOp::Max>();
        case ReductionOp::BitAnd:
            return identity_value_for_type_and_op<T, ReductionOp::BitAnd>();
        case ReductionOp::BitOr:
            return identity_value_for_type_and_op<T, ReductionOp::BitOr>();
        default:
            throw_invalid_reduction_exception(DataType::of<T>(), op);
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
            throw_invalid_reduction_exception(dtype, op);
    }
}

size_t ReductionDef::minimum_destination_bytes_needed() const {
    return checked_mul(data_type.size_in_bytes(), num_outputs);
}

size_t ReductionDef::minimum_source_bytes_needed() const {
    return checked_mul(minimum_destination_bytes_needed(), num_inputs_per_output);
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
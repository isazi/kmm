#include <cstddef>
#include <stdexcept>
#include <utility>

#include "host_operators.hpp"

#include "kmm/memops/types.hpp"
#include "kmm/utils/checked_math.hpp"

namespace kmm {

size_t CopyDef::minimum_source_bytes_needed() const {
    size_t result = src_offset;

    for (size_t i = 0; i < MAX_DIMS; i++) {
        if (counts[i] < 1) {
            return 0;
        }

        result += checked_mul(counts[i] - 1, src_strides[i]);
    }

    return result + element_size;
}

size_t CopyDef::minimum_destination_bytes_needed() const {
    size_t result = dst_offset;

    for (size_t i = 0; i < MAX_DIMS; i++) {
        if (counts[i] < 1) {
            return 0;
        }

        result += checked_mul(counts[i] - 1, dst_strides[i]);
    }

    return result + element_size;
}

void CopyDef::add_dimension(
    size_t count,
    size_t src_offset,
    size_t dst_offset,
    size_t src_stride,
    size_t dst_stride
) {
    for (size_t i = 0; i < MAX_DIMS; i++) {
        if (counts[i] == 1) {
            this->src_offset += src_offset * src_stride * element_size;
            this->dst_offset += dst_offset * dst_stride * element_size;

            counts[i] = count;
            src_strides[i] = src_stride * element_size;
            dst_strides[i] = dst_stride * element_size;
            return;
        }
    }

    throw std::length_error("the number of dimensions of a copy operation cannot exceed 3");
}

size_t CopyDef::effective_dimensionality() const {
    for (size_t n = MAX_DIMS; n > 0; n--) {
        if (counts[n - 1] != 1) {
            return n;
        }
    }

    return 0;
}

size_t CopyDef::number_of_bytes_copied() const {
    return checked_mul(checked_product(counts, counts + MAX_DIMS), element_size);
}

void CopyDef::simplify() {
    if (number_of_bytes_copied() == 0) {
        element_size = 0;
        src_offset = 0;
        dst_offset = 0;

        for (size_t i = 0; i < MAX_DIMS; i++) {
            counts[i] = 0;
            src_strides[i] = 1;
            dst_strides[i] = 1;
        }

        return;
    }

    for (size_t i = 0; i < MAX_DIMS; i++) {
        for (size_t j = 0; j < MAX_DIMS; j++) {
            if (src_strides[j] == element_size && dst_strides[j] == element_size) {
                element_size *= counts[j];
                counts[j] = 1;
                src_strides[j] = 1;
                dst_strides[j] = 1;
            }
        }
    }

    for (size_t i = 0; i < MAX_DIMS; i++) {
        for (size_t j = 0; j < MAX_DIMS; j++) {
            if (i != j && src_strides[j] == counts[i] * src_strides[i]
                && dst_strides[j] == counts[i] * dst_strides[i]) {
                counts[i] *= counts[j];

                counts[j] = 1;
                src_strides[j] = 1;
                dst_strides[j] = 1;
            }
        }
    }

    for (size_t i = 0; i < MAX_DIMS; i++) {
        if (counts[i] == 1) {
            src_strides[i] = 0;
            dst_strides[i] = 0;
        }
    }

    for (size_t i = 0; i < MAX_DIMS; i++) {
        for (size_t j = i + 1; j < MAX_DIMS; j++) {
            if ((counts[i] == 1 && counts[j] != 1) || dst_strides[i] > dst_strides[j]
                || (dst_strides[i] == dst_strides[j] && src_strides[i] > src_strides[j])) {
                std::swap(counts[i], counts[j]);
                std::swap(src_strides[i], src_strides[j]);
                std::swap(dst_strides[i], dst_strides[j]);
            }
        }
    }

    for (size_t i = 0; i < MAX_DIMS; i++) {
        if (counts[i] == 1) {
            if (i == 0) {
                src_strides[0] = element_size;
                dst_strides[0] = element_size;
            } else {
                src_strides[i] = src_strides[i - 1];
                dst_strides[i] = dst_strides[i - 1];
            }
        }
    }
}

size_t FillDef::minimum_destination_bytes_needed() const {
    return checked_mul(checked_add(offset_elements, num_elements), fill_value.size());
}

size_t ReductionDef::minimum_destination_bytes_needed() const {
    return checked_mul(data_type.size_in_bytes(), output_offset_elements + num_outputs);
}

size_t ReductionDef::minimum_source_bytes_needed() const {
    return checked_mul(
        data_type.size_in_bytes(),
        input_offset_elements + checked_mul(num_inputs_per_output, input_stride_elements)
    );
}

[[noreturn]] void throw_invalid_reduction_exception(DataType dtype, Reduction op) {
    throw std::runtime_error(fmt::format("invalid reduction {} for type {}", op, dtype));
}

template<typename T, Reduction Op>
std::vector<uint8_t> identity_value_for_type_and_op() {
    if constexpr (IsReductionSupported<T, Op>()) {
        T value = ReductionOperator<T, Op>::identity();

        uint8_t buffer[sizeof(T)];
        ::memcpy(buffer, &value, sizeof(T));
        return {buffer, buffer + sizeof(T)};
    } else {
        throw_invalid_reduction_exception(DataType::of<T>(), Op);
    }
}

template<typename T>
std::vector<uint8_t> identity_value_for_type(Reduction op) {
    switch (op) {
        case Reduction::Sum:
            return identity_value_for_type_and_op<T, Reduction::Sum>();
        case Reduction::Product:
            return identity_value_for_type_and_op<T, Reduction::Product>();
        case Reduction::Min:
            return identity_value_for_type_and_op<T, Reduction::Min>();
        case Reduction::Max:
            return identity_value_for_type_and_op<T, Reduction::Max>();
        case Reduction::BitAnd:
            return identity_value_for_type_and_op<T, Reduction::BitAnd>();
        case Reduction::BitOr:
            return identity_value_for_type_and_op<T, Reduction::BitOr>();
        default:
            throw_invalid_reduction_exception(DataType::of<T>(), op);
    }
}

std::vector<uint8_t> reduction_identity_value(DataType dtype, Reduction op) {
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

}  // namespace kmm
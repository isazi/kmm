#include "host_reducers.hpp"

#include "kmm/memops/host_reduction.hpp"
#include "kmm/utils/macros.hpp"

namespace kmm {

template<size_t NumRows, typename T, ReductionOp Op>
KMM_NOINLINE void execute_reduction_few_rows(
    const T* __restrict__ src_buffer,
    T* __restrict__ dst_buffer,
    size_t num_columns
) {
    for (size_t j = 0; j < num_columns; j++) {
        dst_buffer[j] = src_buffer[j];

#pragma unroll
        for (size_t i = 1; i < NumRows; i++) {
            dst_buffer[j] =
                ReductionFunctor<T, Op>::combine(dst_buffer[j], src_buffer[i * num_columns + j]);
        }
    }
}

template<typename T, ReductionOp Op>
KMM_NOINLINE void execute_reduction_basic(
    const T* __restrict__ src_buffer,
    T* __restrict__ dst_buffer,
    size_t num_columns,
    size_t num_rows
) {
    for (size_t j = 0; j < num_columns; j++) {
        dst_buffer[j] = src_buffer[j];
    }

    for (size_t i = 1; i < num_rows; i++) {
        for (size_t j = 0; j < num_columns; j++) {
            dst_buffer[j] =
                ReductionFunctor<T, Op>::combine(dst_buffer[j], src_buffer[i * num_columns + j]);
        }
    }
}

template<typename T, ReductionOp Op>
KMM_NOINLINE void execute_reduction_impl(
    const T* src_buffer,
    T* dst_buffer,
    size_t num_columns,
    size_t num_rows
) {
    // For zero rows, we just fill with the identity value.
    if (num_rows == 0) {
        std::fill_n(dst_buffer, num_columns, ReductionFunctor<T, Op>::identity());
        return;
    }

#define KMM_IMPL_REDUCTION_CASE(N)                                                          \
    if (num_rows == (N)) {                                                                  \
        return execute_reduction_few_rows<(N), T, Op>(src_buffer, dst_buffer, num_columns); \
    }

    // Specialize based on the number of rows
    KMM_IMPL_REDUCTION_CASE(1)
    KMM_IMPL_REDUCTION_CASE(2)
    KMM_IMPL_REDUCTION_CASE(3)
    KMM_IMPL_REDUCTION_CASE(4)
    KMM_IMPL_REDUCTION_CASE(5)
    KMM_IMPL_REDUCTION_CASE(6)
    KMM_IMPL_REDUCTION_CASE(7)
    KMM_IMPL_REDUCTION_CASE(8)

    return execute_reduction_basic<T, Op>(src_buffer, dst_buffer, num_columns, num_rows);
}

// NOLINTNEXTLINE
void execute_reduction(const void* src_buffer, void* dst_buffer, Reduction reduction) {
    // NOLINTNEXTLINE
#define KMM_CALL_REDUCTION_FOR_TYPE_AND_OP(T, OP)                                  \
    if constexpr (ReductionFunctorSupported<T, ReductionOp::OP>()) {               \
        if (reduction.operation == ReductionOp::OP) {                              \
            execute_reduction_impl<T, ReductionOp::OP>(                            \
                static_cast<const T*>(src_buffer) + reduction.src_offset_elements, \
                static_cast<T*>(dst_buffer) + reduction.dst_offset_elements,       \
                reduction.num_outputs,                                             \
                reduction.num_inputs_per_output                                    \
            );                                                                     \
            return;                                                                \
        }                                                                          \
    }

    // NOLINTNEXTLINE
#define KMM_CALL_REDUCTION_FOR_TYPE(T)             \
    KMM_CALL_REDUCTION_FOR_TYPE_AND_OP(T, Sum)     \
    KMM_CALL_REDUCTION_FOR_TYPE_AND_OP(T, Product) \
    KMM_CALL_REDUCTION_FOR_TYPE_AND_OP(T, Min)     \
    KMM_CALL_REDUCTION_FOR_TYPE_AND_OP(T, Max)     \
    KMM_CALL_REDUCTION_FOR_TYPE_AND_OP(T, BitAnd)  \
    KMM_CALL_REDUCTION_FOR_TYPE_AND_OP(T, BitOr)

    // NOLINTNEXTLINE
    switch (reduction.data_type.get()) {
        case ScalarKind::Int8:
            KMM_CALL_REDUCTION_FOR_TYPE(int8_t)
            break;
        case ScalarKind::Int16:
            KMM_CALL_REDUCTION_FOR_TYPE(int16_t)
            break;
        case ScalarKind::Int32:
            KMM_CALL_REDUCTION_FOR_TYPE(int32_t)
            break;
        case ScalarKind::Int64:
            KMM_CALL_REDUCTION_FOR_TYPE(int64_t)
            break;
        case ScalarKind::Uint8:
            KMM_CALL_REDUCTION_FOR_TYPE(uint8_t)
            break;
        case ScalarKind::Uint16:
            KMM_CALL_REDUCTION_FOR_TYPE(uint16_t)
            break;
        case ScalarKind::Uint32:
            KMM_CALL_REDUCTION_FOR_TYPE(uint32_t)
            break;
        case ScalarKind::Uint64:
            KMM_CALL_REDUCTION_FOR_TYPE(uint64_t)
            break;
        case ScalarKind::Float32:
            KMM_CALL_REDUCTION_FOR_TYPE(float)
            break;
        case ScalarKind::Float64:
            KMM_CALL_REDUCTION_FOR_TYPE(double)
            break;
        case ScalarKind::KeyAndInt64:
            KMM_CALL_REDUCTION_FOR_TYPE(KeyValue<int64_t>)
            break;
        case ScalarKind::KeyAndFloat64:
            KMM_CALL_REDUCTION_FOR_TYPE(KeyValue<double>)
            break;
        default:
            break;
    }

    throw std::runtime_error("unsupported reduction operation");
}

}  // namespace kmm
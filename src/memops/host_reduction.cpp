#include "kmm/memops/host_reduction.hpp"
#include "kmm/utils/macros.hpp"

namespace kmm {

template<size_t NumRows, typename T, typename F>
KMM_NOINLINE void execute_reduction_few_rows(
    const T* __restrict__ src_buffer,
    T* __restrict__ dst_buffer,
    size_t num_columns,
    F reduce) {
    for (size_t j = 0; j < num_columns; j++) {
#pragma unroll
        for (size_t i = 0; i < NumRows; i++) {
            dst_buffer[j] = reduce(dst_buffer[j], src_buffer[i * num_columns + j]);
        }
    }
}

template<typename T, typename F>
KMM_NOINLINE void execute_reduction_basic(
    const T* __restrict__ src_buffer,
    T* __restrict__ dst_buffer,
    size_t num_columns,
    size_t num_rows,
    F reduce) {
    for (size_t i = 0; i < num_rows; i++) {
        for (size_t j = 0; j < num_columns; j++) {
            dst_buffer[j] = reduce(dst_buffer[j], src_buffer[i * num_columns + j]);
        }
    }
}

template<typename T, typename F>
KMM_NOINLINE void execute_reduction_impl(
    const T* src_buffer,
    T* dst_buffer,
    size_t num_columns,
    size_t num_rows,
    F reduce) {
#define KMM_IMPL_REDUCTION_CASE(N)                                                           \
    if (num_rows == (N)) {                                                                   \
        return execute_reduction_few_rows<(N)>(src_buffer, dst_buffer, num_columns, reduce); \
    }

    // Specialize based on the number of rows
    KMM_IMPL_REDUCTION_CASE(1)
    KMM_IMPL_REDUCTION_CASE(2)
    KMM_IMPL_REDUCTION_CASE(3)
    KMM_IMPL_REDUCTION_CASE(4)
    KMM_IMPL_REDUCTION_CASE(5)
    KMM_IMPL_REDUCTION_CASE(6)
    KMM_IMPL_REDUCTION_CASE(7)

    return execute_reduction_basic(src_buffer, dst_buffer, num_columns, num_rows, reduce);
}

// NOLINTNEXTLINE
void execute_reduction(const void* src_buffer, void* dst_buffer, Reduction reduction) {
    // NOLINTNEXTLINE
#define KMM_IMPL_REDUCTION(T, OP, FUN)            \
    if (reduction.operation == ReductionOp::OP) { \
        execute_reduction_impl(                   \
            static_cast<const T*>(src_buffer),    \
            static_cast<T*>(dst_buffer),          \
            reduction.num_outputs,                \
            reduction.num_inputs_per_output,      \
            [](T a, T b) { return (FUN); });      \
        return;                                   \
    }

    // NOLINTNEXTLINE
#define KMM_IMPL_REDUCTION_ARITHMETIC(T)        \
    KMM_IMPL_REDUCTION(T, Sum, a + b)           \
    KMM_IMPL_REDUCTION(T, Product, (a * b))     \
    KMM_IMPL_REDUCTION(T, Min, (a < b ? a : b)) \
    KMM_IMPL_REDUCTION(T, Max, (a > b ? a : b))

    // NOLINTNEXTLINE
#define KMM_IMPL_REDUCTION_INTEGER(T)   \
    KMM_IMPL_REDUCTION_ARITHMETIC(T)    \
    KMM_IMPL_REDUCTION(T, BitAnd, a& b) \
    KMM_IMPL_REDUCTION(T, BitOr, a | b)

    // NOLINTNEXTLINE
    switch (reduction.data_type.get()) {
        case ScalarKind::Int8:
            KMM_IMPL_REDUCTION_INTEGER(int8_t)
            break;
        case ScalarKind::Int16:
            KMM_IMPL_REDUCTION_INTEGER(int16_t)
            break;
        case ScalarKind::Int32:
            KMM_IMPL_REDUCTION_INTEGER(int32_t)
            break;
        case ScalarKind::Int64:
            KMM_IMPL_REDUCTION_INTEGER(int64_t)
            break;
        case ScalarKind::Uint8:
            KMM_IMPL_REDUCTION_INTEGER(uint8_t)
            break;
        case ScalarKind::Uint16:
            KMM_IMPL_REDUCTION_INTEGER(uint16_t)
            break;
        case ScalarKind::Uint32:
            KMM_IMPL_REDUCTION_INTEGER(uint32_t)
            break;
        case ScalarKind::Uint64:
            KMM_IMPL_REDUCTION_INTEGER(uint64_t)
            break;
        case ScalarKind::Float32:
            KMM_IMPL_REDUCTION_ARITHMETIC(float)
            break;
        case ScalarKind::Float64:
            KMM_IMPL_REDUCTION_ARITHMETIC(double)
            break;
        default:
            break;
    }

    throw std::runtime_error("unsupported reduction operation");
}

}  // namespace kmm
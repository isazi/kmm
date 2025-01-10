#pragma once

#include <type_traits>
#include <utility>

#include "host_operators.hpp"

#include "kmm/core/backends.hpp"
#include "kmm/memops/types.hpp"

namespace kmm {

template<typename F, typename = void>
struct GPUAtomic;

template<typename M, typename T, typename F>
KMM_DEVICE void gpu_generic_atomicCAS(T* output, T input, F combine) {
    static_assert(sizeof(T) == sizeof(M));
    static_assert(alignof(T) >= alignof(M));

    M old_bits = *reinterpret_cast<M*>(output);
    M assumed_bits;
    M new_bits;

    do {
        assumed_bits = old_bits;

        T old_value;
        ::memcpy(&old_value, &old_bits, sizeof(T));

        T new_value = combine(old_value, input);
        ::memcpy(&new_bits, &new_value, sizeof(T));

        if (assumed_bits == new_bits) {
            break;
        }

        old_bits = atomicCAS(reinterpret_cast<M*>(output), assumed_bits, new_bits);
    } while (old_bits != assumed_bits);
}

template<typename T, Reduction Op>
struct GPUAtomic<ReductionOperator<T, Op>, std::enable_if_t<sizeof(T) == 2 && alignof(T) == 2>> {
    static KMM_DEVICE void atomic_combine(T* output, T input) {
        gpu_generic_atomicCAS<unsigned short>(output, input, ReductionOperator<T, Op>());
    }
};

template<typename T, Reduction Op>
struct GPUAtomic<ReductionOperator<T, Op>, std::enable_if_t<sizeof(T) == 4 && alignof(T) == 4>> {
    static KMM_DEVICE void atomic_combine(T* output, T input) {
        gpu_generic_atomicCAS<unsigned int>(output, input, ReductionOperator<T, Op>());
    }
};

template<typename T, Reduction Op>
struct GPUAtomic<ReductionOperator<T, Op>, std::enable_if_t<sizeof(T) == 8 && alignof(T) == 8>> {
    static KMM_DEVICE void atomic_combine(T* output, T input) {
        gpu_generic_atomicCAS<unsigned long long int>(output, input, ReductionOperator<T, Op>());
    }
};

#define KMM_GPU_ATOMIC_REDUCTION_IMPL(T, OP, EXPR)                  \
    template<>                                                      \
    struct GPUAtomic<ReductionOperator<T, OP>> {                    \
        static KMM_DEVICE void atomic_combine(T* output, T input) { \
            EXPR(output, input);                                    \
        }                                                           \
    };

KMM_GPU_ATOMIC_REDUCTION_IMPL(int, Reduction::BitAnd, atomicAnd)
KMM_GPU_ATOMIC_REDUCTION_IMPL(long long int, Reduction::BitAnd, atomicAnd)
KMM_GPU_ATOMIC_REDUCTION_IMPL(unsigned int, Reduction::BitAnd, atomicAnd)
KMM_GPU_ATOMIC_REDUCTION_IMPL(unsigned long long int, Reduction::BitAnd, atomicAnd)

KMM_GPU_ATOMIC_REDUCTION_IMPL(int, Reduction::BitOr, atomicOr)
KMM_GPU_ATOMIC_REDUCTION_IMPL(long long int, Reduction::BitOr, atomicOr)
KMM_GPU_ATOMIC_REDUCTION_IMPL(unsigned int, Reduction::BitOr, atomicOr)
KMM_GPU_ATOMIC_REDUCTION_IMPL(unsigned long long int, Reduction::BitOr, atomicOr)

KMM_GPU_ATOMIC_REDUCTION_IMPL(double, Reduction::Sum, atomicAdd)
KMM_GPU_ATOMIC_REDUCTION_IMPL(float, Reduction::Sum, atomicAdd)
KMM_GPU_ATOMIC_REDUCTION_IMPL(int, Reduction::Sum, atomicAdd)
KMM_GPU_ATOMIC_REDUCTION_IMPL(unsigned int, Reduction::Sum, atomicAdd)
//KMM_GPU_ATOMIC_REDUCTION_IMPL(long long int, ReductionOp::Sum, atomicAdd)
KMM_GPU_ATOMIC_REDUCTION_IMPL(unsigned long long int, Reduction::Sum, atomicAdd)
KMM_GPU_ATOMIC_REDUCTION_IMPL(half_type, Reduction::Sum, atomicAdd)
KMM_GPU_ATOMIC_REDUCTION_IMPL(bfloat16_type, Reduction::Sum, atomicAdd)

//KMM_GPU_ATOMIC_REDUCTION_IMPL(double, ReductionOp::Min, atomicMin)
//KMM_GPU_ATOMIC_REDUCTION_IMPL(float, ReductionOp::Min, atomicMin)
KMM_GPU_ATOMIC_REDUCTION_IMPL(int, Reduction::Min, atomicMin)
KMM_GPU_ATOMIC_REDUCTION_IMPL(long long int, Reduction::Min, atomicMin)
KMM_GPU_ATOMIC_REDUCTION_IMPL(unsigned int, Reduction::Min, atomicMin)
KMM_GPU_ATOMIC_REDUCTION_IMPL(long long unsigned int, Reduction::Min, atomicMin)
//KMM_GPU_ATOMIC_REDUCTION_IMPL(__half, ReductionOp::Min, atomicMin)
//KMM_GPU_ATOMIC_REDUCTION_IMPL(__nv_bfloat16, ReductionOp::Min, atomicMin)

//KMM_GPU_ATOMIC_REDUCTION_IMPL(double, ReductionOp::Max, atomicMax)
//KMM_GPU_ATOMIC_REDUCTION_IMPL(float, ReductionOp::Max, atomicMax)
KMM_GPU_ATOMIC_REDUCTION_IMPL(int, Reduction::Max, atomicMax)
KMM_GPU_ATOMIC_REDUCTION_IMPL(unsigned int, Reduction::Max, atomicMax)
KMM_GPU_ATOMIC_REDUCTION_IMPL(long long int, Reduction::Max, atomicMax)
KMM_GPU_ATOMIC_REDUCTION_IMPL(unsigned long long int, Reduction::Max, atomicMax)
//KMM_GPU_ATOMIC_REDUCTION_IMPL(__half, ReductionOp::Max, atomicMax)
//KMM_GPU_ATOMIC_REDUCTION_IMPL(__nv_bfloat16, ReductionOp::Max, atomicMax)

template<typename F, typename = void>
struct IsGPUAtomicSupported: std::false_type {};

template<typename F>
struct IsGPUAtomicSupported<F, std::void_t<decltype(GPUAtomic<F>())>>: std::true_type {};

}  // namespace kmm

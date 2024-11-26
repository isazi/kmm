#pragma once

#include "kmm/memops/host_reducers.hpp"
#include "kmm/core/reduction.hpp"

namespace kmm {

template<typename T, ReductionOp Op, typename = void>
struct GPUReductionFunctor;

template<typename M, ReductionOp Op, typename T>
KMM_DEVICE void gpu_generic_cas(T* output, T input) {
    static_assert(sizeof(T) == sizeof(M));
    static_assert(alignof(T) >= alignof(M));

    M old_bits = *reinterpret_cast<M*>(output);
    M assumed_bits;
    M new_bits;

    do {
        assumed_bits = old_bits;

        T old_value;
        ::memcpy(&old_value, &old_bits, sizeof(T));

        T new_value = ReductionFunctor<T, Op>::combine(old_value, input);
        ::memcpy(&new_bits, &new_value, sizeof(T));

        if (assumed_bits == new_bits) {
            break;
        }

        old_bits = atomicCAS(reinterpret_cast<M*>(output), assumed_bits, new_bits);
    } while (old_bits != assumed_bits);
}

template<typename T, ReductionOp Op>
struct GPUReductionFunctor<
    T,
    Op,
    std::enable_if_t<ReductionFunctorSupported<T, Op>() && sizeof(T) == 2 && alignof(T) == 2>>:
    ReductionFunctor<T, Op> {
    static KMM_DEVICE void atomic_combine(T* output, T input) {
        gpu_generic_cas<unsigned short, Op>(output, input);
    }
};

template<typename T, ReductionOp Op>
struct GPUReductionFunctor<
    T,
    Op,
    std::enable_if_t<ReductionFunctorSupported<T, Op>() && sizeof(T) == 4 && alignof(T) == 4>>:
    ReductionFunctor<T, Op> {
    static KMM_DEVICE void atomic_combine(T* output, T input) {
        gpu_generic_cas<unsigned int, Op>(output, input);
    }
};

template<typename T, ReductionOp Op>
struct GPUReductionFunctor<
    T,
    Op,
    std::enable_if_t<ReductionFunctorSupported<T, Op>() && sizeof(T) == 8 && alignof(T) == 8>>:
    ReductionFunctor<T, Op> {
    static KMM_DEVICE void atomic_combine(T* output, T input) {
        gpu_generic_cas<unsigned long long int, Op>(output, input);
    }
};

#define GPU_REDUCTION_IMPL(T, OP, EXPR)                            \
    template<>                                                      \
    struct GPUReductionFunctor<T, OP>: ReductionFunctor<T, OP> {   \
        static KMM_DEVICE void atomic_combine(T* output, T input) { \
            EXPR(output, input);                                    \
        }                                                           \
    };

GPU_REDUCTION_IMPL(int, ReductionOp::BitAnd, atomicAnd)
GPU_REDUCTION_IMPL(long long int, ReductionOp::BitAnd, atomicAnd)
GPU_REDUCTION_IMPL(unsigned int, ReductionOp::BitAnd, atomicAnd)
GPU_REDUCTION_IMPL(unsigned long long int, ReductionOp::BitAnd, atomicAnd)

GPU_REDUCTION_IMPL(int, ReductionOp::BitOr, atomicOr)
GPU_REDUCTION_IMPL(long long int, ReductionOp::BitOr, atomicOr)
GPU_REDUCTION_IMPL(unsigned int, ReductionOp::BitOr, atomicOr)
GPU_REDUCTION_IMPL(unsigned long long int, ReductionOp::BitOr, atomicOr)

GPU_REDUCTION_IMPL(double, ReductionOp::Sum, atomicAdd)
GPU_REDUCTION_IMPL(float, ReductionOp::Sum, atomicAdd)
GPU_REDUCTION_IMPL(int, ReductionOp::Sum, atomicAdd)
GPU_REDUCTION_IMPL(unsigned int, ReductionOp::Sum, atomicAdd)
//GPU_REDUCTION_IMPL(long long int, ReductionOp::Sum, atomicAdd)
GPU_REDUCTION_IMPL(unsigned long long int, ReductionOp::Sum, atomicAdd)
GPU_REDUCTION_IMPL(half_type, ReductionOp::Sum, atomicAdd)
GPU_REDUCTION_IMPL(bfloat16_type, ReductionOp::Sum, atomicAdd)

//GPU_REDUCTION_IMPL(double, ReductionOp::Min, atomicMin)
//GPU_REDUCTION_IMPL(float, ReductionOp::Min, atomicMin)
GPU_REDUCTION_IMPL(int, ReductionOp::Min, atomicMin)
GPU_REDUCTION_IMPL(long long int, ReductionOp::Min, atomicMin)
GPU_REDUCTION_IMPL(unsigned int, ReductionOp::Min, atomicMin)
GPU_REDUCTION_IMPL(long long unsigned int, ReductionOp::Min, atomicMin)
//GPU_REDUCTION_IMPL(__half, ReductionOp::Min, atomicMin)
//GPU_REDUCTION_IMPL(__nv_bfloat16, ReductionOp::Min, atomicMin)

//GPU_REDUCTION_IMPL(double, ReductionOp::Max, atomicMax)
//GPU_REDUCTION_IMPL(float, ReductionOp::Max, atomicMax)
GPU_REDUCTION_IMPL(int, ReductionOp::Max, atomicMax)
GPU_REDUCTION_IMPL(unsigned int, ReductionOp::Max, atomicMax)
GPU_REDUCTION_IMPL(long long int, ReductionOp::Max, atomicMax)
GPU_REDUCTION_IMPL(unsigned long long int, ReductionOp::Max, atomicMax)
//GPU_REDUCTION_IMPL(__half, ReductionOp::Max, atomicMax)
//GPU_REDUCTION_IMPL(__nv_bfloat16, ReductionOp::Max, atomicMax)

template<typename T, ReductionOp Op, typename = void>
struct GPUReductionFunctorSupported: std::false_type {};

template<typename T, ReductionOp Op>
struct GPUReductionFunctorSupported<T, Op, std::void_t<decltype(GPUReductionFunctor<T, Op>())>>:
    std::true_type {};

}  // namespace kmm

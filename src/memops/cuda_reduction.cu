#include <float.h>

#include "kmm/memops/cuda_reduction.hpp"
#include "kmm/utils/checked_math.hpp"
#include "kmm/utils/cuda.hpp"
#include "kmm/utils/integer_fun.hpp"

namespace kmm {

template <typename T, ReductionOp Op>
struct ReductionFun;

template <typename T, ReductionOp Op, typename = void>
struct ReductionFunSupported: std::false_type {};

template <typename T, ReductionOp Op>
struct ReductionFunSupported<T, Op, std::void_t<decltype(ReductionFun<T, Op>())>>: std::true_type {

};

template <>
struct ReductionFun<float, ReductionOp::Sum> {
    static __device__ float identity() {
        return 0;
    }

    static __device__ float combine(float a, float b) {
        return a + b;
    }

    static __device__ void atomic_combine(float* result, float value) {
        atomicAdd(result, value);
    }
};

template <>
struct ReductionFun<int32_t, ReductionOp::Min> {
    static constexpr int32_t MIN_VALUE = std::numeric_limits<int32_t>::min();

    static __device__ int32_t identity() {
        return MIN_VALUE;
    }

    static __device__ int32_t combine(int32_t a, int32_t b) {
        return a < b ? a : b;
    }

    static __device__ void atomic_combine(int32_t* result, int32_t value) {
        atomicMin(result, value);
    }
};

static constexpr size_t total_block_size = 256;

template <typename T, ReductionOp Op>
__global__ void reduction_kernel(
    const T* src_buffer,
    T* dst_buffer,
    size_t num_outputs,
    size_t num_partials_per_output,
    size_t partials_per_thread
) {
    __shared__ T shared_results[total_block_size];

    uint64_t thread_x = threadIdx.x;
    uint64_t thread_y = threadIdx.y;

    uint64_t global_x = blockIdx.x * uint64_t(blockDim.x) + thread_x;
    uint64_t global_y = blockIdx.y * uint64_t(blockDim.y) + thread_y;

    T local_result = ReductionFun<T, Op>::identity();

    for (size_t i = 0; i < partials_per_thread; i++) {
        size_t x = global_x;
        size_t y = global_y * partials_per_thread + i;

        if (x < num_outputs && y < partials_per_thread) {
            T partial_result = src_buffer[y * num_partials_per_output + x];
            local_result = ReductionFun<T, Op>::combine(local_result, partial_result);
        } else {
            break;
        }
    }

    shared_results[thread_y * blockDim.x + thread_x] = local_result;

    __syncthreads();

    if (thread_y == 0) {
        for (size_t y = 1; y < blockDim.y; y++) {
            T partial_result = shared_results[y * blockDim.x + global_x];
            local_result = ReductionFun<T, Op>::combine(local_result, partial_result);
        }

        if (global_x < num_outputs) {
            ReductionFun<T, Op>::atomic_combine(&dst_buffer[global_x], local_result);
        }
    }
}

template <typename T, ReductionOp Op>
void execute_reduction_for_type_and_op(
    CUstream stream,
    CUdeviceptr src_buffer,
    CUdeviceptr dst_buffer,
    size_t num_outputs,
    size_t num_partials_per_output
) {
    if constexpr (ReductionFunSupported<T, Op>()) {
        size_t block_size_x = 1;
        size_t block_size_y = total_block_size;

        while (block_size_x < num_outputs && block_size_y > 1) {
            block_size_x *= 2;
            block_size_y /= 2;
        }

        // Divide the total number of elements by the total number of threads
        //  - Total elements: #outputs x #partials-per-output
        //  - Total threads: #threads-per-block x #blocks-per-gpu
        // We use max 512 blocks on the GPU as a rough heuristic here
        size_t partials_per_thread = div_ceil(
            num_outputs * num_partials_per_output,
            block_size_x * block_size_y * 512
        );

        dim3 block_size = {
            checked_cast<unsigned int>(block_size_x),
            checked_cast<unsigned int>(block_size_y),
        };

        dim3 grid_size = {
            checked_cast<unsigned int>(div_ceil(num_outputs, block_size_x)),
            checked_cast<unsigned int>(div_ceil(num_partials_per_output, block_size_y * partials_per_thread))
        };

        reduction_kernel<T, Op><<<grid_size, block_size, 0, stream>>>(
            reinterpret_cast<const T*>(src_buffer),
            reinterpret_cast<T*>(dst_buffer),
            num_outputs,
            num_partials_per_output,
            partials_per_thread
        );

        KMM_CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error(fmt::format("reduction {} for data type {} is not yet supported",
                                             Op, DataType::of<T>()));
    }
}

template <typename T>
void execute_reduction_for_type(
    CUstream stream,
    ReductionOp operation,
    CUdeviceptr src_buffer,
    CUdeviceptr dst_buffer,
    size_t num_outputs,
    size_t num_partials_per_output
) {
#define KMM_CALL_REDUCTION_FOR_TYPE_AND_OP(O) \
    execute_reduction_for_type_and_op<T, O>(     \
        stream, src_buffer, dst_buffer, num_outputs, num_partials_per_output);

    switch (operation) {
        case ReductionOp::Sum:
            KMM_CALL_REDUCTION_FOR_TYPE_AND_OP(ReductionOp::Sum)
            break;
        case ReductionOp::Product:
            KMM_CALL_REDUCTION_FOR_TYPE_AND_OP(ReductionOp::Product)
            break;
        case ReductionOp::Min:
            KMM_CALL_REDUCTION_FOR_TYPE_AND_OP(ReductionOp::Min)
            break;
        case ReductionOp::Max:
            KMM_CALL_REDUCTION_FOR_TYPE_AND_OP(ReductionOp::Max)
            break;
        case ReductionOp::BitAnd:
            KMM_CALL_REDUCTION_FOR_TYPE_AND_OP(ReductionOp::BitAnd)
            break;
        case ReductionOp::BitOr:
            KMM_CALL_REDUCTION_FOR_TYPE_AND_OP(ReductionOp::BitOr)
            break;
        default:
            throw std::runtime_error(fmt::format("reductions for operation {} are not yet supported",
                                                 operation));
    }
}

void execute_cuda_reduction_async(
    CUstream stream,
    CUdeviceptr src_buffer,
    CUdeviceptr dst_buffer,
    Reduction reduction) {
#define KMM_CALL_REDUCTION_FOR_TYPE(T) \
    execute_reduction_for_type<T>(     \
        stream, reduction.operation, src_buffer, dst_buffer, reduction.num_outputs, reduction.num_inputs_per_output);

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
        default:
            throw std::runtime_error(fmt::format("reductions on data type {} are not yet supported",
                                                 reduction.data_type));
    }

}
}
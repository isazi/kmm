#include <float.h>

#include "kmm/memops/gpu_fill.hpp"
#include "kmm/memops/gpu_reduction.hpp"
#include "kmm/memops/gpu_reducers.hpp"
#include "kmm/utils/checked_math.hpp"
#include "kmm/utils/gpu.hpp"
#include "kmm/utils/integer_fun.hpp"

namespace kmm {

static constexpr size_t total_block_size = 256;

#ifdef KMM_USE_DEVICE
template<typename T, ReductionOp Op, bool UseAtomics = true>
__global__ void reduction_kernel(
    const T* src_buffer,
    T* dst_buffer,
    size_t num_outputs,
    size_t num_inputs_per_output,
    size_t items_per_thread
) {
    __shared__ T shared_results[total_block_size];

    uint32_t thread_x = threadIdx.x;
    uint32_t thread_y = threadIdx.y;

    uint64_t global_x = blockIdx.x * uint64_t(blockDim.x) + thread_x;
    uint64_t global_y = blockIdx.y * uint64_t(blockDim.y) * items_per_thread + thread_y;

    T local_result = ReductionFunctor<T, Op>::identity();

    if (global_x < num_outputs && global_y < num_inputs_per_output) {
        size_t x = global_x;
        size_t max_y = min(global_y + items_per_thread * blockDim.y, num_inputs_per_output);

        for (size_t y = global_y; y < max_y; y += blockDim.y) {
            T partial_result = src_buffer[y * num_outputs + x];
            local_result = ReductionFunctor<T, Op>::combine(local_result, partial_result);
        }

        shared_results[thread_y * blockDim.x + thread_x] = local_result;
    }

    __syncthreads();

    if (global_x < num_outputs && thread_y == 0) {
        for (unsigned int y = 1; y < blockDim.y; y++) {
            T partial_result = shared_results[y * blockDim.x + thread_x];
            local_result = ReductionFunctor<T, Op>::combine(local_result, partial_result);
        }

        if constexpr (UseAtomics) {
            GPUReductionFunctor<T, Op>::atomic_combine(&dst_buffer[global_x], local_result);
        } else {
            dst_buffer[global_x] = local_result;
        }
    }
}
#endif

template<typename T, ReductionOp Op>
void execute_reduction_for_type_and_op(
    stream_t stream,
    GPUdeviceptr src_buffer,
    GPUdeviceptr dst_buffer,
    size_t num_outputs,
    size_t num_partials_per_output
) {
    size_t block_size_x;
    size_t block_size_y;
    size_t items_per_thread;

    if (num_partials_per_output <= 8) {
        block_size_x = total_block_size;
        block_size_y = 1;
        items_per_thread = num_partials_per_output;
    } else if (num_outputs < 32) {
        block_size_x = round_up_to_power_of_two(num_outputs);
        block_size_y = total_block_size / block_size_x;
        items_per_thread = 1;
    } else {
        block_size_x = 32;
        block_size_y = total_block_size / block_size_x;
        items_per_thread = 8;
    }

    // Divide the total number of elements by the total number of threads
    // We use max 512 blocks on the GPU as a rough heuristic here
    size_t max_blocks_per_gpu = 512;
    size_t max_grid_size_y = div_ceil(max_blocks_per_gpu, div_ceil(num_outputs, block_size_x));

    // If we do not have atomics, we can only have 1 block in the y-direction
    if (!GPUReductionFunctorSupported<T, Op>()) {
        max_grid_size_y = 1;
    }

    // The minimum items per thread is the number of partials divided by the maximum threads along Y
    size_t min_items_per_thread = div_ceil(num_partials_per_output, max_grid_size_y * block_size_y);

    if (min_items_per_thread > items_per_thread) {
        items_per_thread = min_items_per_thread;
    }

    dim3 block_size = {
        checked_cast<unsigned int>(block_size_x),
        checked_cast<unsigned int>(block_size_y),
    };

    dim3 grid_size = {
        checked_cast<unsigned int>(div_ceil(num_outputs, block_size_x)),
        checked_cast<unsigned int>(
            div_ceil(num_partials_per_output, block_size_y * items_per_thread)
        )};

    if (grid_size.y == 1) {
        if constexpr (ReductionFunctorSupported<T, Op>()) {
#ifdef KMM_USE_DEVICE
            reduction_kernel<T, Op, false><<<grid_size, block_size, 0, stream>>>(
                reinterpret_cast<const T*>(src_buffer),
                reinterpret_cast<T*>(dst_buffer),
                num_outputs,
                num_partials_per_output,
                items_per_thread
            );
#endif

            KMM_GPU_CHECK(gpuGetLastError());
            return;
        }
    }

    if constexpr (GPUReductionFunctorSupported<T, Op>()) {
        T identity = ReductionFunctor<T, Op>::identity();
        execute_gpu_fill_async(stream, dst_buffer, num_outputs * sizeof(T), &identity, sizeof(T));

#ifdef KMM_USE_DEVICE
        reduction_kernel<T, Op><<<grid_size, block_size, 0, stream>>>(
            reinterpret_cast<const T*>(src_buffer),
            reinterpret_cast<T*>(dst_buffer),
            num_outputs,
            num_partials_per_output,
            items_per_thread
        );
#endif

        KMM_GPU_CHECK(gpuGetLastError());
        return;
    }

    // silence unused warnings
    (void)stream;
    (void)src_buffer;
    (void)dst_buffer;
    (void)block_size;

    throw std::runtime_error(
        fmt::format("reduction {} for data type {} is not yet supported", Op, DataType::of<T>())
    );
}

template<typename T>
void execute_reduction_for_type(
    stream_t stream,
    ReductionOp operation,
    GPUdeviceptr src_buffer,
    GPUdeviceptr dst_buffer,
    size_t num_outputs,
    size_t num_partials_per_output
) {
#define KMM_CALL_REDUCTION_FOR_TYPE_AND_OP(O) \
    execute_reduction_for_type_and_op<T, O>(  \
        stream,                               \
        src_buffer,                           \
        dst_buffer,                           \
        num_outputs,                          \
        num_partials_per_output               \
    );

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
            throw std::runtime_error(
                fmt::format("reductions for operation {} are not yet supported", operation)
            );
    }
}

void execute_gpu_reduction_async(
    stream_t stream,
    GPUdeviceptr src_buffer,
    GPUdeviceptr dst_buffer,
    ReductionDef reduction
) {
    auto element_size = reduction.data_type.size_in_bytes();

#define KMM_CALL_REDUCTION_FOR_TYPE(T)                             \
    execute_reduction_for_type<T>(                                 \
        stream,                                                    \
        reduction.operation,                                       \
        src_buffer + reduction.src_offset_elements * element_size, \
        dst_buffer + reduction.dst_offset_elements * element_size, \
        reduction.num_outputs,                                     \
        reduction.num_inputs_per_output                            \
    );

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
            throw std::runtime_error(
                fmt::format("reductions on data type {} are not yet supported", reduction.data_type)
            );
    }
}
}  // namespace kmm
#pragma once

#include "kmm/internals/backends.hpp"
#include "kmm/core/reduction.hpp"

namespace kmm {

/**
 *
 */
void execute_gpu_reduction_async(
    stream_t stream,
    GPUdeviceptr src_buffer,
    GPUdeviceptr dst_buffer,
    ReductionDef reduction
);

}  // namespace kmm
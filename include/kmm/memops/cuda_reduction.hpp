#pragma once

#include <cuda.h>

#include "kmm/core/reduction.hpp"

namespace kmm {

/**
 *
 */
void execute_cuda_reduction_async(
    CUstream stream,
    CUdeviceptr src_buffer,
    CUdeviceptr dst_buffer,
    Reduction reduction
);

}  // namespace kmm
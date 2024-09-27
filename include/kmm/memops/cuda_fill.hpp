#pragma once

#include <cuda.h>

namespace kmm {

void execute_cuda_fill_async(
    CUstream stream,
    CUdeviceptr dst_buffer,
    size_t nbytes,
    const void* pattern,
    size_t pattern_nbytes);

}
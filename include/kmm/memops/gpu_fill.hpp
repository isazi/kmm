#pragma once

#include "kmm/internals/backends.hpp"

namespace kmm {

void execute_gpu_fill_async(
    stream_t stream,
    GPUdeviceptr dst_buffer,
    size_t nbytes,
    const void* pattern,
    size_t pattern_nbytes
);

}
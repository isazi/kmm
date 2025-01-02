#pragma once

#include "kmm/core/backends.hpp"
#include "kmm/memops/types.hpp"

namespace kmm {

void execute_gpu_fill_async(GPUstream_t stream, GPUdeviceptr dst_buffer, const FillDef& fill);

}
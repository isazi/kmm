#pragma once

#include "kmm/core/fill_def.hpp"
#include "kmm/internals/backends.hpp"

namespace kmm {

void execute_gpu_fill_async(stream_t stream, GPUdeviceptr dst_buffer, const FillDef& fill);

}
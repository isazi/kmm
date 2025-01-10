#include "kmm/core/backends.hpp"
#include "kmm/memops/types.hpp"

namespace kmm {

void execute_gpu_h2d_copy_async(
    GPUstream_t stream,
    const void* src_buffer,
    GPUdeviceptr dst_buffer,
    CopyDef copy_description
);

void execute_gpu_d2h_copy_async(
    GPUstream_t stream,
    GPUdeviceptr src_buffer,
    void* dst_buffer,
    CopyDef copy_description
);

void execute_gpu_d2d_copy_async(
    GPUstream_t stream,
    GPUdeviceptr src_buffer,
    GPUdeviceptr dst_buffer,
    CopyDef copy_description
);

void execute_gpu_h2d_copy(
    const void* src_buffer,
    GPUdeviceptr dst_buffer,
    CopyDef copy_description
);

void execute_gpu_d2h_copy(GPUdeviceptr src_buffer, void* dst_buffer, CopyDef copy_description);

void execute_gpu_d2d_copy(
    GPUdeviceptr src_buffer,
    GPUdeviceptr dst_buffer,
    CopyDef copy_description
);

}  // namespace kmm
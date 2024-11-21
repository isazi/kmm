#include <cuda.h>

#include "kmm/core/copy_def.hpp"

namespace kmm {

void execute_cuda_h2d_copy_async(
    CUstream stream,
    const void* src_buffer,
    CUdeviceptr dst_buffer,
    CopyDef copy_description
);

void execute_cuda_d2h_copy_async(
    CUstream stream,
    CUdeviceptr src_buffer,
    void* dst_buffer,
    CopyDef copy_description
);

void execute_cuda_d2d_copy_async(
    CUstream stream,
    CUdeviceptr src_buffer,
    CUdeviceptr dst_buffer,
    CopyDef copy_description
);

void execute_cuda_h2d_copy(
    const void* src_buffer,
    CUdeviceptr dst_buffer,
    CopyDef copy_description
);

void execute_cuda_d2h_copy(CUdeviceptr src_buffer, void* dst_buffer, CopyDef copy_description);

void execute_cuda_d2d_copy(
    CUdeviceptr src_buffer,
    CUdeviceptr dst_buffer,
    CopyDef copy_description
);

}  // namespace kmm
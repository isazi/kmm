#include <cuda.h>

#include "kmm/core/copy_description.hpp"

namespace kmm {

void execute_cuda_h2d_copy_async(
    CUstream stream,
    const void* src_buffer,
    void* dst_buffer,
    CopyDescription copy_description);

void execute_cuda_d2h_copy_async(
    CUstream stream,
    const void* src_buffer,
    void* dst_buffer,
    CopyDescription copy_description);

void execute_cuda_d2d_copy_async(
    CUstream stream,
    const void* src_buffer,
    void* dst_buffer,
    CopyDescription copy_description);

}  // namespace kmm
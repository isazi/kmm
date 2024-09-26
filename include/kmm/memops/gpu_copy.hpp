
#include "kmm/internals/backends.hpp"
#include "kmm/core/copy_description.hpp"

namespace kmm {

void execute_gpu_h2d_copy_async(
    stream_t stream,
    const void* src_buffer,
    void* dst_buffer,
    CopyDescription copy_description);

void execute_gpu_d2h_copy_async(
    stream_t stream,
    const void* src_buffer,
    void* dst_buffer,
    CopyDescription copy_description);

void execute_gpu_d2d_copy_async(
    stream_t stream,
    const void* src_buffer,
    void* dst_buffer,
    CopyDescription copy_description);

void execute_gpu_h2d_copy(
    const void* src_buffer,
    void* dst_buffer,
    CopyDescription copy_description);

void execute_gpu_d2h_copy(
    const void* src_buffer,
    void* dst_buffer,
    CopyDescription copy_description);

void execute_gpu_d2d_copy(
    const void* src_buffer,
    void* dst_buffer,
    CopyDescription copy_description);

}  // namespace kmm
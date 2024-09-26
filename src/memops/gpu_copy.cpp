
#include <stdexcept>

#include "fmt/format.h"

#include "kmm/memops/gpu_copy.hpp"
#include "kmm/utils/checked_math.hpp"
#include "kmm/utils/gpu.hpp"

namespace kmm {

void throw_unsupported_dimension_exception(size_t dim) {
    throw std::runtime_error(fmt::format(
        "copy operation is {} dimensional, only 1D or 2D copy operations are supported",
        dim + 1));
}

void execute_gpu_h2d_copy_impl(
    std::optional<stream_t> stream,
    const void* src_buffer,
    void* dst_buffer,
    CopyDescription copy_description) {
    copy_description.simplify();
    size_t dim = copy_description.effective_dimensionality();

    GPUdeviceptr dst_ptr = GPUdeviceptr(dst_buffer) + copy_description.dst_offset;
    const void* src_ptr = static_cast<const uint8_t*>(src_buffer) + copy_description.src_offset;

    if (dim == 0) {
        if (stream) {
            KMM_GPU_CHECK(
                gpuMemcpyHtoDAsync(dst_ptr, src_ptr, copy_description.element_size, *stream));
        } else {
            KMM_GPU_CHECK(gpuMemcpyHtoD(dst_ptr, src_ptr, copy_description.element_size));
        }
    } else if (dim == 1) {
        GPU_MEMCPY2D info;
        ::bzero(&info, sizeof(GPU_MEMCPY2D));

        info.srcMemoryType = GPU_MEMORYTYPE_HOST;
        info.srcHost = src_ptr;
        info.srcPitch = checked_cast<unsigned int>(copy_description.src_strides[0]);
        info.dstMemoryType = GPU_MEMORYTYPE_DEVICE;
        info.dstDevice = dst_ptr;
        info.dstPitch = checked_cast<unsigned int>(copy_description.dst_strides[0]);
        info.WidthInBytes = checked_cast<unsigned int>(copy_description.element_size);
        info.Height = checked_cast<unsigned int>(copy_description.counts[0]);

        if (stream) {
            KMM_GPU_CHECK(gpuMemcpy2DAsync(&info, *stream));
        } else {
            KMM_GPU_CHECK(gpuMemcpy2D(&info));
        }
    } else {
        throw_unsupported_dimension_exception(dim);
    }
}

void execute_gpu_d2h_copy_impl(
    std::optional<stream_t> stream,
    const void* src_buffer,
    void* dst_buffer,
    CopyDescription copy_description) {
    copy_description.simplify();
    size_t dim = copy_description.effective_dimensionality();

    void* dst_ptr = static_cast<uint8_t*>(dst_buffer) + copy_description.dst_offset;
    GPUdeviceptr src_ptr = GPUdeviceptr(src_buffer) + copy_description.src_offset;

    if (dim == 0) {
        if (stream) {
            KMM_GPU_CHECK(
                gpuMemcpyDtoHAsync(dst_ptr, src_ptr, copy_description.element_size, *stream));
        } else {
            KMM_GPU_CHECK(gpuMemcpyDtoH(dst_ptr, src_ptr, copy_description.element_size));
        }
    } else if (dim == 1) {
        GPU_MEMCPY2D info;
        ::bzero(&info, sizeof(GPU_MEMCPY2D));

        info.srcMemoryType = GPU_MEMORYTYPE_DEVICE;
        info.srcDevice = src_ptr;
        info.srcPitch = checked_cast<unsigned int>(copy_description.src_strides[0]);
        info.dstMemoryType = GPU_MEMORYTYPE_HOST;
        info.dstHost = dst_ptr;
        info.dstPitch = checked_cast<unsigned int>(copy_description.dst_strides[0]);
        info.WidthInBytes = checked_cast<unsigned int>(copy_description.element_size);
        info.Height = checked_cast<unsigned int>(copy_description.counts[0]);

        if (stream) {
            KMM_GPU_CHECK(gpuMemcpy2DAsync(&info, *stream));
        } else {
            KMM_GPU_CHECK(gpuMemcpy2D(&info));
        }
    } else {
        throw_unsupported_dimension_exception(dim);
    }
}

void execute_gpu_d2d_copy_impl(
    std::optional<stream_t> stream,
    const void* src_buffer,
    void* dst_buffer,
    CopyDescription copy_description) {
    copy_description.simplify();
    size_t dim = copy_description.effective_dimensionality();

    GPUdeviceptr dst_ptr = GPUdeviceptr(dst_buffer) + copy_description.dst_offset;
    GPUdeviceptr src_ptr = GPUdeviceptr(src_buffer) + copy_description.src_offset;

    if (dim == 0) {
        if (stream) {
            KMM_GPU_CHECK(
                gpuMemcpyDtoDAsync(dst_ptr, src_ptr, copy_description.element_size, *stream));
        } else {
            KMM_GPU_CHECK(gpuMemcpyDtoD(dst_ptr, src_ptr, copy_description.element_size));
        }
    } else if (dim == 1) {
        GPU_MEMCPY2D info;
        ::bzero(&info, sizeof(GPU_MEMCPY2D));

        info.srcMemoryType = GPU_MEMORYTYPE_DEVICE;
        info.srcDevice = src_ptr;
        info.srcPitch = checked_cast<unsigned int>(copy_description.src_strides[0]);
        info.dstMemoryType = GPU_MEMORYTYPE_DEVICE;
        info.dstDevice = dst_ptr;
        info.dstPitch = checked_cast<unsigned int>(copy_description.dst_strides[0]);
        info.WidthInBytes = checked_cast<unsigned int>(copy_description.element_size);
        info.Height = checked_cast<unsigned int>(copy_description.counts[0]);

        if (stream) {
            KMM_GPU_CHECK(gpuMemcpy2DAsync(&info, *stream));
        } else {
            KMM_GPU_CHECK(gpuMemcpy2D(&info));
        }
    } else {
        throw_unsupported_dimension_exception(dim);
    }
}

void execute_gpu_h2d_copy(
    const void* src_buffer,
    void* dst_buffer,
    CopyDescription copy_description) {
    execute_gpu_h2d_copy_impl(std::nullopt, src_buffer, dst_buffer, copy_description);
}

void execute_gpu_h2d_copy_async(
    stream_t stream,
    const void* src_buffer,
    void* dst_buffer,
    CopyDescription copy_description) {
    execute_gpu_h2d_copy_impl(stream, src_buffer, dst_buffer, copy_description);
}

void execute_gpu_d2h_copy(
    const void* src_buffer,
    void* dst_buffer,
    CopyDescription copy_description) {
    execute_gpu_d2h_copy_impl(std::nullopt, src_buffer, dst_buffer, copy_description);
}

void execute_gpu_d2h_copy_async(
    stream_t stream,
    const void* src_buffer,
    void* dst_buffer,
    CopyDescription copy_description) {
    execute_gpu_d2h_copy_impl(stream, src_buffer, dst_buffer, copy_description);
}

void execute_gpu_d2d_copy(
    const void* src_buffer,
    void* dst_buffer,
    CopyDescription copy_description) {
    execute_gpu_d2d_copy_impl(std::nullopt, src_buffer, dst_buffer, copy_description);
}

void execute_gpu_d2d_copy_async(
    stream_t stream,
    const void* src_buffer,
    void* dst_buffer,
    CopyDescription copy_description) {
    execute_gpu_d2d_copy_impl(stream, src_buffer, dst_buffer, copy_description);
}
}  // namespace kmm
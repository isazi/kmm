
#include <stdexcept>

#include "fmt/format.h"

#include "kmm/memops/cuda_copy.hpp"
#include "kmm/utils/checked_math.hpp"
#include "kmm/utils/cuda.hpp"

namespace kmm {

void throw_unsupported_dimension_exception(size_t dim) {
    throw std::runtime_error(fmt::format(
        "copy operation is {} dimensional, only 1D or 2D copy operations are supported",
        dim + 1
    ));
}

void execute_cuda_h2d_copy_impl(
    std::optional<CUstream> stream,
    const void* src_buffer,
    CUdeviceptr dst_buffer,
    CopyDef copy_description
) {
    copy_description.simplify();
    size_t dim = copy_description.effective_dimensionality();

    CUdeviceptr dst_ptr = dst_buffer + copy_description.dst_offset;
    const void* src_ptr = static_cast<const uint8_t*>(src_buffer) + copy_description.src_offset;

    if (dim == 0) {
        if (stream) {
            KMM_CUDA_CHECK(cuMemcpyHtoDAsync(  //
                dst_ptr,
                src_ptr,
                copy_description.element_size,
                *stream
            ));
        } else {
            KMM_CUDA_CHECK(cuMemcpyHtoD(  //
                dst_ptr,
                src_ptr,
                copy_description.element_size
            ));
        }
    } else if (dim == 1) {
        CUDA_MEMCPY2D info;
        ::bzero(&info, sizeof(CUDA_MEMCPY2D));

        info.srcMemoryType = CUmemorytype::CU_MEMORYTYPE_HOST;
        info.srcHost = src_ptr;
        info.srcPitch = checked_cast<unsigned int>(copy_description.src_strides[0]);
        info.dstMemoryType = CUmemorytype ::CU_MEMORYTYPE_DEVICE;
        info.dstDevice = dst_ptr;
        info.dstPitch = checked_cast<unsigned int>(copy_description.dst_strides[0]);
        info.WidthInBytes = checked_cast<unsigned int>(copy_description.element_size);
        info.Height = checked_cast<unsigned int>(copy_description.counts[0]);

        if (stream) {
            KMM_CUDA_CHECK(cuMemcpy2DAsync(&info, *stream));
        } else {
            KMM_CUDA_CHECK(cuMemcpy2D(&info));
        }
    } else {
        throw_unsupported_dimension_exception(dim);
    }
}

void execute_cuda_d2h_copy_impl(
    std::optional<CUstream> stream,
    CUdeviceptr src_buffer,
    void* dst_buffer,
    CopyDef copy_description
) {
    copy_description.simplify();
    size_t dim = copy_description.effective_dimensionality();

    void* dst_ptr = static_cast<uint8_t*>(dst_buffer) + copy_description.dst_offset;
    CUdeviceptr src_ptr = src_buffer + copy_description.src_offset;

    if (dim == 0) {
        if (stream) {
            KMM_CUDA_CHECK(cuMemcpyDtoHAsync(  //
                dst_ptr,
                src_ptr,
                copy_description.element_size,
                *stream
            ));
        } else {
            KMM_CUDA_CHECK(cuMemcpyDtoH(  //
                dst_ptr,
                src_ptr,
                copy_description.element_size
            ));
        }
    } else if (dim == 1) {
        CUDA_MEMCPY2D info;
        ::bzero(&info, sizeof(CUDA_MEMCPY2D));

        info.srcMemoryType = CUmemorytype::CU_MEMORYTYPE_DEVICE;
        info.srcDevice = src_ptr;
        info.srcPitch = checked_cast<unsigned int>(copy_description.src_strides[0]);
        info.dstMemoryType = CUmemorytype ::CU_MEMORYTYPE_HOST;
        info.dstHost = dst_ptr;
        info.dstPitch = checked_cast<unsigned int>(copy_description.dst_strides[0]);
        info.WidthInBytes = checked_cast<unsigned int>(copy_description.element_size);
        info.Height = checked_cast<unsigned int>(copy_description.counts[0]);

        if (stream) {
            KMM_CUDA_CHECK(cuMemcpy2DAsync(&info, *stream));
        } else {
            KMM_CUDA_CHECK(cuMemcpy2D(&info));
        }
    } else {
        throw_unsupported_dimension_exception(dim);
    }
}

void execute_cuda_d2d_copy_impl(
    std::optional<CUstream> stream,
    CUdeviceptr src_buffer,
    CUdeviceptr dst_buffer,
    CopyDef copy_description
) {
    copy_description.simplify();
    size_t dim = copy_description.effective_dimensionality();

    CUdeviceptr dst_ptr = dst_buffer + copy_description.dst_offset;
    CUdeviceptr src_ptr = src_buffer + copy_description.src_offset;

    if (dim == 0) {
        if (stream) {
            KMM_CUDA_CHECK(cuMemcpyDtoDAsync(  //
                dst_ptr,
                src_ptr,
                copy_description.element_size,
                *stream
            ));
        } else {
            KMM_CUDA_CHECK(cuMemcpyDtoD(  //
                dst_ptr,
                src_ptr,
                copy_description.element_size
            ));
        }
    } else if (dim == 1) {
        CUDA_MEMCPY2D info;
        ::bzero(&info, sizeof(CUDA_MEMCPY2D));

        info.srcMemoryType = CUmemorytype::CU_MEMORYTYPE_DEVICE;
        info.srcDevice = src_ptr;
        info.srcPitch = checked_cast<unsigned int>(copy_description.src_strides[0]);
        info.dstMemoryType = CUmemorytype ::CU_MEMORYTYPE_DEVICE;
        info.dstDevice = dst_ptr;
        info.dstPitch = checked_cast<unsigned int>(copy_description.dst_strides[0]);
        info.WidthInBytes = checked_cast<unsigned int>(copy_description.element_size);
        info.Height = checked_cast<unsigned int>(copy_description.counts[0]);

        if (stream) {
            KMM_CUDA_CHECK(cuMemcpy2DAsync(&info, *stream));
        } else {
            KMM_CUDA_CHECK(cuMemcpy2D(&info));
        }
    } else {
        throw_unsupported_dimension_exception(dim);
    }
}

void execute_cuda_h2d_copy(
    const void* src_buffer,
    CUdeviceptr dst_buffer,
    CopyDef copy_description
) {
    execute_cuda_h2d_copy_impl(std::nullopt, src_buffer, dst_buffer, copy_description);
}

void execute_cuda_h2d_copy_async(
    CUstream stream,
    const void* src_buffer,
    CUdeviceptr dst_buffer,
    CopyDef copy_description
) {
    execute_cuda_h2d_copy_impl(stream, src_buffer, dst_buffer, copy_description);
}

void execute_cuda_d2h_copy(CUdeviceptr src_buffer, void* dst_buffer, CopyDef copy_description) {
    execute_cuda_d2h_copy_impl(std::nullopt, src_buffer, dst_buffer, copy_description);
}

void execute_cuda_d2h_copy_async(
    CUstream stream,
    CUdeviceptr src_buffer,
    void* dst_buffer,
    CopyDef copy_description
) {
    execute_cuda_d2h_copy_impl(stream, src_buffer, dst_buffer, copy_description);
}

void execute_cuda_d2d_copy(
    CUdeviceptr src_buffer,
    CUdeviceptr dst_buffer,
    CopyDef copy_description
) {
    execute_cuda_d2d_copy_impl(std::nullopt, src_buffer, dst_buffer, copy_description);
}

void execute_cuda_d2d_copy_async(
    CUstream stream,
    CUdeviceptr src_buffer,
    CUdeviceptr dst_buffer,
    CopyDef copy_description
) {
    execute_cuda_d2d_copy_impl(stream, src_buffer, dst_buffer, copy_description);
}
}  // namespace kmm
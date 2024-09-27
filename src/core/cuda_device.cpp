#include <stdexcept>

#include "fmt/format.h"
#include "spdlog/spdlog.h"

#include "kmm/core/cuda_device.hpp"
#include "kmm/memops/cuda_fill.hpp"
#include "kmm/utils/checked_math.hpp"

namespace kmm {

CudaDevice::CudaDevice(CudaDeviceInfo info, CudaContextHandle context, CUstream stream) :
    CudaDeviceInfo(info),
    m_context(context),
    m_stream(stream) {
    CudaContextGuard guard {m_context};

    KMM_CUDA_CHECK(cublasCreate(&m_cublas_handle));
    KMM_CUDA_CHECK(cublasSetStream(m_cublas_handle, m_stream));
}

CudaDevice::~CudaDevice() {
    CudaContextGuard guard {m_context};
    KMM_CUDA_CHECK(cublasDestroy(m_cublas_handle));
}

void CudaDevice::synchronize() const {
    CudaContextGuard guard {m_context};
    KMM_CUDA_CHECK(cuStreamSynchronize(nullptr));
    KMM_CUDA_CHECK(cuStreamSynchronize(m_stream));
}

void CudaDevice::fill_bytes(
    void* dest_buffer,
    size_t nbytes,
    const void* fill_pattern,
    size_t fill_pattern_size) const {
    CudaContextGuard guard {m_context};
    execute_cuda_fill_async(
        m_stream,
        (CUdeviceptr)dest_buffer,
        nbytes,
        fill_pattern,
        fill_pattern_size);
}

void CudaDevice::copy_bytes(const void* source_buffer, void* dest_buffer, size_t nbytes) const {
    CudaContextGuard guard {m_context};
    KMM_CUDA_CHECK(cuMemcpyAsync(
        reinterpret_cast<CUdeviceptr>(dest_buffer),
        reinterpret_cast<CUdeviceptr>(source_buffer),
        nbytes,
        m_stream));
}

}  // namespace kmm
#include <stdexcept>

#include "fmt/format.h"
#include "spdlog/spdlog.h"

#include "kmm/core/gpu_device.hpp"
#include "kmm/utils/checked_math.hpp"

namespace kmm {

GPUDevice::GPUDevice(DeviceInfo info, GPUContextHandle context, stream_t stream) :
    DeviceInfo(info),
    m_context(context),
    m_stream(stream) {
    GPUContextGuard guard {m_context};

    KMM_GPU_CHECK(blasCreate(&m_blas_handle));
    KMM_GPU_CHECK(blasSetStream(m_blas_handle, m_stream));
}

GPUDevice::~GPUDevice() {
    GPUContextGuard guard {m_context};
    KMM_GPU_CHECK(blasDestroy(m_blas_handle));
}

void GPUDevice::synchronize() const {
    GPUContextGuard guard {m_context};
    KMM_GPU_CHECK(gpuStreamSynchronize(nullptr));
    KMM_GPU_CHECK(gpuStreamSynchronize(m_stream));
}

template<size_t N>
bool is_fill_pattern_repetitive(const void* fill_pattern, size_t fill_pattern_size) {
    if (fill_pattern_size % N != 0) {
        return false;
    }

    for (size_t i = 1; i < fill_pattern_size / N; i++) {
        for (size_t j = 0; j < N; j++) {
            if (static_cast<const uint8_t*>(fill_pattern)[i * N + j]
                != static_cast<const uint8_t*>(fill_pattern)[j]) {
                return false;
            }
        }
    }

    return true;
}

void GPUDevice::fill_bytes(
    void* dest_buffer,
    size_t nbytes,
    const void* fill_pattern,
    size_t fill_pattern_size) const {
    GPUContextGuard guard {m_context};
    if (nbytes == 0 || fill_pattern_size == 0) {
        return;
    }

    size_t remainder = nbytes % fill_pattern_size;
    if (remainder != 0) {
        fill_bytes(
            static_cast<uint8_t*>(dest_buffer) + (nbytes - remainder),
            remainder,
            fill_pattern,
            remainder);

        nbytes -= remainder;
    }

    if (is_fill_pattern_repetitive<1>(fill_pattern, fill_pattern_size)) {
        uint8_t pattern;
        ::memcpy(&pattern, fill_pattern, sizeof(uint8_t));
        spdlog::debug("fill async: {} {} {} {}", dest_buffer, pattern, nbytes, (void*)m_stream);
        KMM_GPU_CHECK(gpuMemsetD8Async(GPUdeviceptr(dest_buffer), pattern, nbytes, m_stream));
    } else if (is_fill_pattern_repetitive<2>(fill_pattern, fill_pattern_size)) {
        uint16_t pattern;
        ::memcpy(&pattern, fill_pattern, sizeof(uint16_t));
        KMM_GPU_CHECK(gpuMemsetD16Async(
            GPUdeviceptr(dest_buffer),
            pattern,
            nbytes / sizeof(uint16_t),
            m_stream));
    } else if (is_fill_pattern_repetitive<4>(fill_pattern, fill_pattern_size)) {
        uint32_t pattern;
        ::memcpy(&pattern, fill_pattern, sizeof(uint32_t));
        KMM_GPU_CHECK(gpuMemsetD32Async(
            GPUdeviceptr(dest_buffer),
            pattern,
            nbytes / sizeof(uint32_t),
            m_stream));
    } else {
        throw GPUException(fmt::format(
            "could not fill buffer, value is {} bit, but only 8, 16, or 32 bit is supported",
            fill_pattern_size * 8));
    }
}

void GPUDevice::copy_bytes(const void* source_buffer, void* dest_buffer, size_t nbytes) const {
    GPUContextGuard guard {m_context};
    KMM_GPU_CHECK(gpuMemcpyAsync(
        reinterpret_cast<GPUdeviceptr>(dest_buffer),
        reinterpret_cast<GPUdeviceptr>(const_cast<void *>(source_buffer)),
        nbytes,
        m_stream));
}

}  // namespace kmm

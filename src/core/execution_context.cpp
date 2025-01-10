#include <stdexcept>

#include "fmt/format.h"
#include "spdlog/spdlog.h"

#include "kmm/core/execution_context.hpp"
#include "kmm/memops/gpu_fill.hpp"
#include "kmm/utils/checked_math.hpp"

namespace kmm {

InvalidExecutionContext::InvalidExecutionContext(
    const std::type_info& expected,
    const std::type_info& gotten
) {
    m_message = fmt::format(
        "task expected an execution context of type {}, but was executed with type {}",
        expected.name(),
        gotten.name()
    );
}

const char* InvalidExecutionContext::what() const noexcept {
    return m_message.c_str();
}

DeviceContext::DeviceContext(DeviceInfo info, GPUContextHandle context, GPUstream_t stream) :
    DeviceInfo(info),
    m_context(context),
    m_stream(stream) {
    GPUContextGuard guard {m_context};

    KMM_GPU_CHECK(blasCreate(&m_blas_handle));
    KMM_GPU_CHECK(blasSetStream(m_blas_handle, m_stream));
}

DeviceContext::~DeviceContext() {
    GPUContextGuard guard {m_context};
    KMM_GPU_CHECK(blasDestroy(m_blas_handle));
}

void DeviceContext::synchronize() const {
    GPUContextGuard guard {m_context};
    KMM_GPU_CHECK(gpuStreamSynchronize(nullptr));
    KMM_GPU_CHECK(gpuStreamSynchronize(m_stream));
}

void DeviceContext::fill_bytes(
    void* dest_buffer,
    size_t nbytes,
    const void* fill_pattern,
    size_t fill_pattern_size
) const {
    GPUContextGuard guard {m_context};
    execute_gpu_fill_async(
        m_stream,
        reinterpret_cast<GPUdeviceptr>(dest_buffer),
        FillDef(fill_pattern_size, nbytes / fill_pattern_size, fill_pattern)
    );
}

void DeviceContext::copy_bytes(const void* source_buffer, void* dest_buffer, size_t nbytes) const {
    GPUContextGuard guard {m_context};
    KMM_GPU_CHECK(gpuMemcpyAsync(
        reinterpret_cast<GPUdeviceptr>(dest_buffer),
        reinterpret_cast<GPUdeviceptr>(const_cast<void*>(source_buffer)),
        nbytes,
        m_stream
    ));
}

}  // namespace kmm
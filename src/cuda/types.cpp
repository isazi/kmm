#include <utility>

#include "fmt/format.h"
#include "spdlog/spdlog.h"

#include "kmm/cuda/types.hpp"
#include "kmm/panic.hpp"

#ifdef KMM_USE_CUDA

namespace kmm {

void cuda_throw_exception(CUresult result, const char* file, int line, const char* expression) {
    throw CudaDriverException(fmt::format("{} ({}:{})", expression, file, line), result);
}

void cuda_throw_exception(cudaError_t result, const char* file, int line, const char* expression) {
    throw CudaRuntimeException(fmt::format("{} ({}:{})", expression, file, line), result);
}

void cuda_throw_exception(
    cublasStatus_t result,
    const char* file,
    int line,
    const char* expression) {
    throw CudaBlasException(fmt::format("{} ({}:{})", expression, file, line), result);
}

CudaDriverException::CudaDriverException(const std::string& message, CUresult result) :
    status(result) {
    const char* name = "???";
    const char* description = "???";

    // Ignore the return code from these functions
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &description);

    m_message = fmt::format("CUDA driver error: {} ({}): {}", description, name, message);
}

CudaRuntimeException::CudaRuntimeException(const std::string& message, cudaError_t result) :
    status(result) {
    const char* name = "???";
    const char* description = "???";

    // Ignore the return code from these functions
    name = cudaGetErrorName(result);
    description = cudaGetErrorString(result);

    m_message = fmt::format("CUDA runtime error: {} ({}): {}", description, name, message);
}

CudaBlasException::CudaBlasException(const std::string& message, cublasStatus_t result) :
    status(result) {
    const char* name = [&]() {
        switch (result) {
            case CUBLAS_STATUS_SUCCESS:
                return "CUBLAS_STATUS_SUCCESS";
            case CUBLAS_STATUS_NOT_INITIALIZED:
                return "CUBLAS_STATUS_NOT_INITIALIZED";
            case CUBLAS_STATUS_ALLOC_FAILED:
                return "CUBLAS_STATUS_ALLOC_FAILED";
            case CUBLAS_STATUS_INVALID_VALUE:
                return "CUBLAS_STATUS_INVALID_VALUE";
            case CUBLAS_STATUS_ARCH_MISMATCH:
                return "CUBLAS_STATUS_ARCH_MISMATCH";
            case CUBLAS_STATUS_MAPPING_ERROR:
                return "CUBLAS_STATUS_MAPPING_ERROR";
            case CUBLAS_STATUS_EXECUTION_FAILED:
                return "CUBLAS_STATUS_EXECUTION_FAILED";
            case CUBLAS_STATUS_INTERNAL_ERROR:
                return "CUBLAS_STATUS_INTERNAL_ERROR";
            case CUBLAS_STATUS_NOT_SUPPORTED:
                return "CUBLAS_STATUS_NOT_SUPPORTED";
            case CUBLAS_STATUS_LICENSE_ERROR:
                return "CUBLAS_STATUS_LICENSE_ERROR";
            default:
                return "???";
        }
    }();

    m_message = fmt::format("CUDA BLAS error: {}: {}", name, message);
}

CudaContextHandle::CudaContextHandle(CUcontext context, std::shared_ptr<void> lifetime) :
    m_context(context),
    m_lifetime(std::move(lifetime)) {}

std::vector<CUdevice> get_cuda_devices() {
    try {
        auto result = cuInit(0);
        if (result == CUDA_ERROR_NO_DEVICE) {
            return {};
        }

        if (result != CUDA_SUCCESS) {
            throw CudaDriverException("cuInit failed", result);
        }

        int count = 0;
        KMM_CUDA_CHECK(cuDeviceGetCount(&count));

        std::vector<CUdevice> devices {count};
        for (int i = 0; i < count; i++) {
            KMM_CUDA_CHECK(cuDeviceGet(&devices[static_cast<size_t>(i)], i));
        }

        return devices;
    } catch (const CudaException& e) {
        spdlog::warn("ignored error while initializing: {}", e.what());
        return {};
    }
}

std::optional<CUdevice> get_cuda_device_by_address(const void* address) {
    try {
        CUmemorytype memory_type;
        KMM_CUDA_CHECK(cuPointerGetAttribute(
            &memory_type,
            CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
            CUdeviceptr(address)));

        if (memory_type == CUmemorytype_enum::CU_MEMORYTYPE_DEVICE) {
            int ordinal;
            KMM_CUDA_CHECK(cuPointerGetAttribute(
                &ordinal,
                CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                CUdeviceptr(address)));

            return CUdevice {ordinal};
        }
    } catch (const std::exception& error) {
        spdlog::warn("ignored error in `get_cuda_device_by_address`: {}", error.what());
    }

    return std::nullopt;
}

CudaContextHandle CudaContextHandle::create_context_for_device(CUdevice device) {
    int flags = CU_CTX_MAP_HOST;
    CUcontext context;
    KMM_CUDA_CHECK(cuCtxCreate(&context, flags, device));

    auto lifetime = std::shared_ptr<void>(nullptr, [=](const void* ignore) {
        KMM_ASSERT(cuCtxDestroy(context) == CUDA_SUCCESS);
    });

    return {context, lifetime};
}

CudaContextHandle CudaContextHandle::retain_primary_context_for_device(CUdevice device) {
    CUcontext context;
    KMM_CUDA_CHECK(cuDevicePrimaryCtxRetain(&context, device));

    auto lifetime = std::shared_ptr<void>(nullptr, [=](const void* ignore) {
        KMM_ASSERT(cuDevicePrimaryCtxRelease(device) == CUDA_SUCCESS);
    });

    return {context, lifetime};
}

CudaContextGuard::CudaContextGuard(CudaContextHandle context) : m_context(std::move(context)) {
    KMM_CUDA_CHECK(cuCtxPushCurrent(m_context));
}

CudaContextGuard::~CudaContextGuard() {
    CUcontext previous;
    KMM_ASSERT(cuCtxPopCurrent(&previous) == CUDA_SUCCESS);
}

}  // namespace kmm

#endif  // KMM_USE_CUDA
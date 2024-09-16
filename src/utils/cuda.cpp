#include "fmt/format.h"
#include "spdlog/spdlog.h"

#include "kmm/utils/cuda.hpp"
#include "kmm/utils/panic.hpp"

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
    const char* name = cublasGetStatusName(result);
    const char* description = cublasGetStatusString(result);

    m_message = fmt::format("cuBLAS runtime error: {} ({}): {}", description, name, message);
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

        std::vector<CUdevice> devices {};
        for (int i = 0; i < count; i++) {
            CUdevice device;
            KMM_CUDA_CHECK(cuDeviceGet(&device, i));
            devices.push_back(device);
        }

        return devices;
    } catch (const CudaException& e) {
        spdlog::warn("ignored error while initializing: {}", e.what());
        return {};
    }
}

std::optional<CUdevice> get_cuda_device_by_address(const void* address) {
    int ordinal;
    CUmemorytype memory_type;
    CUresult result =
        cuPointerGetAttribute(&memory_type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, CUdeviceptr(address));

    if (result == CUDA_SUCCESS && memory_type == CU_MEMORYTYPE_DEVICE) {
        result = cuPointerGetAttribute(
            &ordinal,
            CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
            CUdeviceptr(address));

        if (result == CUDA_SUCCESS) {
            return CUdevice {ordinal};
        }
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
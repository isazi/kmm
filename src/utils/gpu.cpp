#include "fmt/format.h"
#include "spdlog/spdlog.h"

#include "kmm/utils/gpu.hpp"
#include "kmm/utils/panic.hpp"

namespace kmm {

void gpu_throw_exception(GPUresult result, const char* file, int line, const char* expression) {
    throw GPUDriverException(fmt::format("{} ({}:{})", expression, file, line), result);
}

void gpu_throw_exception(gpuError_t result, const char* file, int line, const char* expression) {
    throw GPURuntimeException(fmt::format("{} ({}:{})", expression, file, line), result);
}

void gpu_throw_exception(blasStatus_t result, const char* file, int line, const char* expression) {
    throw BlasException(fmt::format("{} ({}:{})", expression, file, line), result);
}

GPUDriverException::GPUDriverException(const std::string& message, GPUresult result) :
    status(result) {
    const char* name = "???";
    const char* description = "???";

    // Ignore the return code from these functions
    gpuGetErrorName(result, &name);
    gpuGetErrorString(result, &description);

    m_message = fmt::format("GPU driver error: {} ({}): {}", description, name, message);
}

GPURuntimeException::GPURuntimeException(const std::string& message, gpuError_t result) :
    status(result) {
    const char* name = "???";
    const char* description = "???";

    // Ignore the return code from these functions
    name = GPUrtGetErrorName(result);
    description = GPUrtGetErrorString(result);

    m_message = fmt::format("GPU runtime error: {} ({}): {}", description, name, message);
}

BlasException::BlasException(const std::string& message, blasStatus_t result) : status(result) {
    const char* name = blasGetStatusName(result);
    const char* description = blasGetStatusString(result);

    m_message = fmt::format("BLAS runtime error: {} ({}): {}", description, name, message);
}

GPUContextHandle::GPUContextHandle(GPUcontext context, std::shared_ptr<void> lifetime) :
    m_context(context),
    m_lifetime(std::move(lifetime)) {}

std::vector<GPUdevice> get_gpu_devices() {
    try {
        auto result = gpuInit(0);
        if (result == GPU_ERROR_NO_DEVICE) {
            return {};
        }

        if (result != GPU_SUCCESS) {
            throw GPUDriverException("gpuInit failed", result);
        }

        int count = 0;
        KMM_GPU_CHECK(gpuDeviceGetCount(&count));

        std::vector<GPUdevice> devices {};
        for (int i = 0; i < count; i++) {
            GPUdevice device;
            KMM_GPU_CHECK(gpuDeviceGet(&device, i));
            devices.push_back(device);
        }

        return devices;
    } catch (const GPUException& e) {
        spdlog::warn("ignored error while initializing: {}", e.what());
        return {};
    }
}

std::optional<GPUdevice> get_gpu_device_by_address(const void* address) {
    int ordinal;
    GPUmemorytype memory_type;
    GPUresult result = gpuPointerGetAttribute(
        &memory_type,
        GPU_POINTER_ATTRIBUTE_MEMORY_TYPE,
        GPUdeviceptr(address)
    );

    if (result == GPU_SUCCESS && memory_type == GPU_MEMORYTYPE_DEVICE) {
        result = gpuPointerGetAttribute(
            &ordinal,
            GPU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
            GPUdeviceptr(address)
        );

        if (result == GPU_SUCCESS) {
            return GPUdevice {ordinal};
        }
    }

    return std::nullopt;
}

GPUContextHandle GPUContextHandle::create_context_for_device(GPUdevice device) {
    int flags = GPU_CTX_MAP_HOST;
    GPUcontext context;
    KMM_GPU_CHECK(gpuCtxCreate(&context, flags, device));

    auto lifetime = std::shared_ptr<void>(nullptr, [=](const void* ignore) {
        KMM_ASSERT(gpuCtxDestroy(context) == GPU_SUCCESS);
    });

    return {context, lifetime};
}

GPUContextHandle GPUContextHandle::retain_primary_context_for_device(GPUdevice device) {
    GPUcontext context;
    KMM_GPU_CHECK(gpuDevicePrimaryCtxRetain(&context, device));

    auto lifetime = std::shared_ptr<void>(nullptr, [=](const void* ignore) {
        KMM_ASSERT(gpuDevicePrimaryCtxRelease(device) == GPU_SUCCESS);
    });

    return {context, lifetime};
}

GPUContextGuard::GPUContextGuard(GPUContextHandle context) : m_context(std::move(context)) {
    KMM_GPU_CHECK(gpuCtxPushCurrent(m_context));
}

GPUContextGuard::~GPUContextGuard() {
    GPUcontext previous;
    KMM_ASSERT(gpuCtxPopCurrent(&previous) == GPU_SUCCESS);
}

}  // namespace kmm

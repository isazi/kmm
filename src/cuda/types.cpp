#include <utility>

#include "fmt/format.h"

#include "kmm/cuda/types.hpp"
#include "kmm/panic.hpp"

namespace kmm {

void cuda_throw_exception(CUresult result, const char* file, int line, const char* expression) {
    throw CudaException(fmt::format("{} ({}:{})", expression, file, line), result);
}

CudaException::CudaException(const std::string& message, CUresult result) : m_status(result) {
    const char* name = "???";
    const char* description = "???";

    // Ignore the return code from these functions
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &description);

    m_message = fmt::format("CUDA error: {} ({}): {}", description, name, message);
}

CudaContextHandle::CudaContextHandle(CUcontext context, std::shared_ptr<void> lifetime) :
    m_context(context),
    m_lifetime(std::move(lifetime)) {}

CudaContextHandle CudaContextHandle::from_new_context(CUdevice device) {
    int flags = CU_CTX_MAP_HOST;
    CUcontext context;
    KMM_CUDA_CHECK(cuCtxCreate(&context, flags, device));

    auto lifetime = std::shared_ptr<void>(nullptr, [=](const void* ignore) {
        KMM_ASSERT(cuCtxDestroy(context) == CUDA_SUCCESS);
    });

    return {context, lifetime};
}

CudaContextHandle CudaContextHandle::from_primary_context(CUdevice device) {
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
#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "kmm/utils/macros.hpp"

#define KMM_CUDA_CHECK(...)                                                        \
    do {                                                                           \
        auto __code = (__VA_ARGS__);                                               \
        if (__code != decltype(__code)(0)) {                                       \
            ::kmm::cuda_throw_exception(__code, __FILE__, __LINE__, #__VA_ARGS__); \
        }                                                                          \
    } while (0);

namespace kmm {

void cuda_throw_exception(CUresult result, const char* file, int line, const char* expression);
void cuda_throw_exception(cudaError_t result, const char* file, int line, const char* expression);

class CudaException: std::exception {
  public:
    CudaException(std::string message = {}) : m_message(std::move(message)) {}

    const char* what() const noexcept override {
        return m_message.c_str();
    }

  protected:
    std::string m_message;
};

class CudaDriverException: CudaException {
  public:
    CudaDriverException(const std::string& message, CUresult result);
    CudaDriverException(const char* message, CUresult result) :
        CudaDriverException(std::string(message), result) {}
    CUresult status;
};

class CudaRuntimeException: CudaException {
  public:
    CudaRuntimeException(const std::string& message, cudaError_t result);
    cudaError_t status;
};

/**
 * Returns the available CUDA devices as a list of `CUdevice`s.
 */
std::vector<CUdevice> get_cuda_devices();

/**
 * If the given address points to memory allocation that has been allocated on a CUDA device, then
 * this function returns the device ordinal as a `CUdevice`. If the address points ot an invalid
 * memory location or a non-CUDA buffer, then returns `std::nullopt`.
 */
std::optional<CUdevice> get_cuda_device_by_address(const void* address);

class CudaContextHandle {
    CudaContextHandle() = delete;
    CudaContextHandle(CUcontext context, std::shared_ptr<void> lifetime);

  public:
    static CudaContextHandle create_context_for_device(CUdevice device);
    static CudaContextHandle retain_primary_context_for_device(CUdevice device);

    operator CUcontext() const {
        return m_context;
    }

  private:
    CUcontext m_context;
    std::shared_ptr<void> m_lifetime;
};

inline bool operator==(const CudaContextHandle& lhs, const CudaContextHandle& rhs) {
    return CUcontext(lhs) == CUcontext(rhs);
}

inline bool operator!=(const CudaContextHandle& lhs, const CudaContextHandle& rhs) {
    return !(lhs == rhs);
}

class CudaContextGuard {
    KMM_NOT_COPYABLE_OR_MOVABLE(CudaContextGuard)

  public:
    CudaContextGuard(CudaContextHandle context);
    ~CudaContextGuard();

  private:
    CudaContextHandle m_context;
};

}  // namespace kmm
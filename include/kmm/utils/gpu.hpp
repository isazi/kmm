#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "kmm/internals/backends.hpp"
#include "kmm/utils/macros.hpp"

#define KMM_GPU_CHECK(...)                                                        \
    do {                                                                          \
        auto __code = (__VA_ARGS__);                                              \
        if (__code != decltype(__code)(0)) {                                      \
            ::kmm::gpu_throw_exception(__code, __FILE__, __LINE__, #__VA_ARGS__); \
        }                                                                         \
    } while (0);

namespace kmm {

void gpu_throw_exception(GPUresult result, const char* file, int line, const char* expression);
void gpu_throw_exception(gpuError_t result, const char* file, int line, const char* expression);
void gpu_throw_exception(blasStatus_t result, const char* file, int line, const char* expression);

class GPUException: public std::exception {
  public:
    GPUException(std::string message = {}) : m_message(std::move(message)) {}

    const char* what() const noexcept override {
        return m_message.c_str();
    }

  protected:
    std::string m_message;
};

class GPUDriverException: public GPUException {
  public:
    GPUDriverException(const std::string& message, GPUresult result);
    GPUDriverException(const char* message, GPUresult result) :
        GPUDriverException(std::string(message), result) {}
    GPUresult status;
};

class GPURuntimeException: public GPUException {
  public:
    GPURuntimeException(const std::string& message, gpuError_t result);
    gpuError_t status;
};

class BlasException: public GPUException {
  public:
    BlasException(const std::string& message, blasStatus_t result);
    blasStatus_t status;
};

/**
 * Returns the available devices as a list of `device`s.
 */
std::vector<GPUdevice> get_gpu_devices();

/**
 * If the given address points to memory allocation that has been allocated on a GPU, then
 * this function returns the device ordinal as a `device`. If the address points ot an invalid
 * memory location or a non-GPU buffer, then it returns `std::nullopt`.
 */
std::optional<GPUdevice> get_gpu_device_by_address(const void* address);

class GPUContextHandle {
    GPUContextHandle() = delete;
    GPUContextHandle(GPUcontext context, std::shared_ptr<void> lifetime);

  public:
    static GPUContextHandle create_context_for_device(GPUdevice device);
    static GPUContextHandle retain_primary_context_for_device(GPUdevice device);

    operator GPUcontext() const {
        return m_context;
    }

  private:
    GPUcontext m_context;
    std::shared_ptr<void> m_lifetime;
};

inline bool operator==(const GPUContextHandle& lhs, const GPUContextHandle& rhs) {
    return GPUcontext(lhs) == GPUcontext(rhs);
}

inline bool operator!=(const GPUContextHandle& lhs, const GPUContextHandle& rhs) {
    return !(lhs == rhs);
}

class GPUContextGuard {
    KMM_NOT_COPYABLE_OR_MOVABLE(GPUContextGuard)

  public:
    GPUContextGuard(GPUContextHandle context);
    ~GPUContextGuard();

  private:
    GPUContextHandle m_context;
};

}  // namespace kmm

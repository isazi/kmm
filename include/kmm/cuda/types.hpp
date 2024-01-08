#pragma once

#include <cuda.h>
#include <memory>
#include <string>

#define KMM_CUDA_CHECK(...)                                                               \
    do {                                                                                  \
        auto code = (__VA_ARGS__);                                                        \
        if (code != decltype(code)(0)) {                                                  \
            ::kmm::cuda_throw_exception((__VA_ARGS__), __FILE__, __LINE__, #__VA_ARGS__); \
        }                                                                                 \
    } while (0);

namespace kmm {

static constexpr CUstream_st* const CUDA_DEFAULT_STREAM = nullptr;

void cuda_throw_exception(CUresult result, const char* file, int line, const char* expression);

class CudaException: public std::exception {
  public:
    CudaException(const std::string& message, CUresult result = CUDA_ERROR_UNKNOWN);

    CUresult status() const {
        return m_status;
    }

    const char* what() const noexcept override {
        return m_message.c_str();
    }

  private:
    CUresult m_status;
    std::string m_message;
};

class CudaContextHandle {
    CudaContextHandle(CUcontext context, std::shared_ptr<void> lifetime);

  public:
    static CudaContextHandle from_new_context(CUdevice device);
    static CudaContextHandle from_primary_context(CUdevice device);

    CudaContextHandle() = delete;
    CudaContextHandle(const CudaContextHandle&) = default;
    CudaContextHandle(CudaContextHandle&&) noexcept = default;

    CudaContextHandle& operator=(const CudaContextHandle&) = default;
    CudaContextHandle& operator=(CudaContextHandle&&) noexcept = default;

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
  public:
    CudaContextGuard(CudaContextHandle context);
    ~CudaContextGuard();

    CudaContextGuard(const CudaContextGuard&) = delete;
    CudaContextGuard(CudaContextGuard&&) noexcept = delete;

    CudaContextGuard& operator=(const CudaContextGuard&) = delete;
    CudaContextGuard& operator=(CudaContextGuard&&) noexcept = delete;

  private:
    CudaContextHandle m_context;
};

}  // namespace kmm
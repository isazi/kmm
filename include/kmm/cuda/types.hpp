#pragma once

#include <memory>
#include <string>
#include <vector>

#ifdef USE_CUDA
    #include <cuda.h>
#endif

#include "kmm/types.hpp"

<<<<<<< HEAD
#ifdef USE_CUDA
#define KMM_CUDA_CHECK(...)                                                        \
    do {                                                                           \
        auto __code = (__VA_ARGS__);                                               \
        if (__code != decltype(__code)(0)) {                                       \
            ::kmm::cuda_throw_exception(__code, __FILE__, __LINE__, #__VA_ARGS__); \
        }                                                                          \
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

std::vector<CUdevice> get_cuda_devices();

class CudaContextHandle {
    CudaContextHandle() = delete;
    CudaContextHandle(CUcontext context, std::shared_ptr<void> lifetime);

  public:
    static CudaContextHandle from_new_context(CUdevice device);
    static CudaContextHandle from_primary_context(CUdevice device);

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

#endif  // USE_CUDA
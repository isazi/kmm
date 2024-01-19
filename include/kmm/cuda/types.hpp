#pragma once

#include <memory>
#include <string>
#include <vector>

#ifdef KMM_USE_CUDA
    #include <cublas_v2.h>
    #include <cuda.h>
    #include <cuda_runtime_api.h>
#endif

#include "kmm/utils/macros.hpp"

#ifdef KMM_USE_CUDA
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
void cuda_throw_exception(cudaError_t result, const char* file, int line, const char* expression);
void cuda_throw_exception(
    cublasStatus_t result,
    const char* file,
    int line,
    const char* expression);

class CudaException: public std::exception {
  public:
    CudaException(const std::string& message = "") : m_message(message) {}

    const char* what() const noexcept override {
        return m_message.c_str();
    }

  protected:
    std::string m_message;
};

class CudaDriverException: public CudaException {
  public:
    CudaDriverException(const std::string& message, CUresult result);
    CUresult status;
};

class CudaRuntimeException: public CudaException {
  public:
    CudaRuntimeException(const std::string& message, cudaError_t result);
    cudaError_t status;
};

class CudaBlasException: public CudaException {
  public:
    CudaBlasException(const std::string& message, cublasStatus_t status);
    cublasStatus_t status;
};

std::vector<CUdevice> get_cuda_devices();

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

#endif  // KMM_USE_CUDA
#pragma once

#define KMM_NOT_COPYABLE(TYPE)             \
  public:                                  \
    TYPE(const TYPE&) = delete;            \
    TYPE& operator=(const TYPE&) = delete; \
    TYPE(TYPE&) = delete;                  \
    TYPE& operator=(TYPE&) = delete;       \
                                           \
  private:

#define KMM_NOT_COPYABLE_OR_MOVABLE(TYPE)            \
    KMM_NOT_COPYABLE(TYPE)                           \
  public:                                            \
    TYPE(TYPE&&) noexcept = delete;                  \
    TYPE& operator=(TYPE&&) noexcept = delete;       \
    TYPE(const TYPE&&) noexcept = delete;            \
    TYPE& operator=(const TYPE&&) noexcept = delete; \
                                                     \
  private:

#define KMM_INLINE   __attribute__((always_inline)) inline
#define KMM_NOINLINE __attribute__((noinline))

#define KMM_ASSUME(expr)   __builtin_assume(expr)
#define KMM_UNLIKELY(expr) (__builtin_expect(!(expr), false))
#define KMM_LIKELY(expr)   KMM_UNLIKELY(!(expr))

#if defined(__CUDACC__) || defined(__HIPCC__)
    // CUDA or HIP
    #define KMM_HOST_DEVICE          __host__ __device__ __forceinline__
    #define KMM_DEVICE               __device__ __forceinline__
    #define KMM_HOST_DEVICE_NOINLINE __host__ __device__
    #define KMM_DEVICE_NOINLINE      __device__
#else
    // Dummy backend
    #define KMM_HOST_DEVICE KMM_INLINE
    #define KMM_DEVICE      KMM_INLINE
    #define KMM_HOST_DEVICE_NOINLINE
    #define KMM_DEVICE_NOINLINE
#endif
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

#define KMM_INLINE   __attribute__((always_inline))
#define KMM_NOINLINE __attribute__((noinline))

// If we are use the NVIDIA compiler
#ifdef __NVCC__
    #define KMM_HOST_DEVICE __host__ __device__ __forceinline__
    #define KMM_DEVICE      __device__ __forceinline__

    #define KMM_HOST_DEVICE_NOINLINE __host__ __device__
    #define KMM_DEVICE_NOINLINE      __device__
#else
    #define KMM_HOST_DEVICE KMM_INLINE
    #define KMM_DEVICE      KMM_INLINE

    #define KMM_HOST_DEVICE_NOINLINE
    #define KMM_DEVICE_NOINLINE
#endif

#define KMM_ASSUME(expr) __builtin_assume(expr)
#define KMM_EXPECT(expr) (__builtin_expect(!!(expr), true))
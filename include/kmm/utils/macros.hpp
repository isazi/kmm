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
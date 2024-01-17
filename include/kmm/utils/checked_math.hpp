#pragma once

#include <limits>

namespace kmm {

namespace details {
template<typename T>
struct checked_arithmetic_impl;

#define KMM_IMPL_CHECKED_ARITHMETIC(T, ADD_FUN, MUL_FUN)                  \
    template<>                                                            \
    struct checked_arithmetic_impl<T> {                                   \
        static constexpr T MIN = std::numeric_limits<T>::min();           \
        static constexpr T MAX = std::numeric_limits<T>::max();           \
        static constexpr bool SIGNED = std::numeric_limits<T>::is_signed; \
                                                                          \
        static bool add(T lhs, T rhs, T* result) {                        \
            return ADD_FUN(lhs, rhs, result) == false;                    \
        }                                                                 \
                                                                          \
        static bool mul(T lhs, T rhs, T* result) {                        \
            return MUL_FUN(lhs, rhs, result) == false;                    \
        }                                                                 \
    };

KMM_IMPL_CHECKED_ARITHMETIC(signed int, __builtin_sadd_overflow, __builtin_smul_overflow)
KMM_IMPL_CHECKED_ARITHMETIC(signed long, __builtin_saddl_overflow, __builtin_smull_overflow)
KMM_IMPL_CHECKED_ARITHMETIC(signed long long, __builtin_saddll_overflow, __builtin_smulll_overflow)

KMM_IMPL_CHECKED_ARITHMETIC(unsigned int, __builtin_uadd_overflow, __builtin_umul_overflow)
KMM_IMPL_CHECKED_ARITHMETIC(unsigned long, __builtin_uaddl_overflow, __builtin_umull_overflow)
KMM_IMPL_CHECKED_ARITHMETIC(
    unsigned long long,
    __builtin_uaddll_overflow,
    __builtin_umulll_overflow)

template<
    typename L,
    typename R,
    bool = checked_arithmetic_impl<L>::SIGNED,
    bool = checked_arithmetic_impl<R>::SIGNED>
struct checked_compare_impl {
    static bool equals(const L& lhs, const R& rhs) {
        return lhs == rhs;
    }

    static bool less(const L& lhs, const R& rhs) {
        return lhs < rhs;
    }
};

template<typename L, typename R>
struct checked_compare_impl<L, R, true, false> {
    static bool equals(const L& lhs, const R& rhs) {
        return lhs >= 0 && lhs == rhs;
    }

    static bool less(const L& lhs, const R& rhs) {
        return lhs < 0 || lhs < rhs;
    }
};

template<typename L, typename R>
struct checked_compare_impl<L, R, false, true> {
    static bool equals(const L& lhs, const R& rhs) {
        return rhs >= 0 && lhs == rhs;
    }

    static bool less(const L& lhs, const R& rhs) {
        return rhs >= 0 && lhs < rhs;
    }
};

}  // namespace details

[[noreturn]] void throw_overflow_exception();

template<typename T>
T checked_add(const T& lhs, const T& rhs) {
    T result;
    if (!details::checked_arithmetic_impl<T>::add(lhs, rhs, &result)) {
        throw_overflow_exception();
    }

    return result;
}

template<typename T>
T checked_sum(const T* begin, const T* end) {
    T result = static_cast<T>(0);
    bool is_valid = true;

    for (auto* it = begin; it != end; it++) {
        is_valid &= details::checked_arithmetic_impl<T>::add(result, *it, &result);
    }

    if (!is_valid) {
        throw_overflow_exception();
    }

    return result;
}

template<typename T>
T checked_mul(const T& lhs, const T& rhs) {
    T result;
    if (!details::checked_arithmetic_impl<T>::mul(lhs, rhs, &result)) {
        throw_overflow_exception();
    }

    return result;
}

template<typename T>
T checked_product(const T* begin, const T* end) {
    T result = static_cast<T>(1);
    bool is_valid = true;

    for (auto* it = begin; it != end; it++) {
        is_valid &= details::checked_arithmetic_impl<T>::mul(result, *it, &result);
    }

    if (!is_valid) {
        throw_overflow_exception();
    }

    return result;
}

template<typename L, typename R>
bool cmp_equal(const L& lhs, const R& rhs) {
    return details::checked_compare_impl<L, R>::equals(lhs, rhs);
}

template<typename L, typename R>
bool cmp_less(const L& lhs, const R& rhs) {
    return details::checked_compare_impl<L, R>::less(lhs, rhs);
}

template<typename R, typename T>
R checked_cast(const T& input) {
    if (cmp_less(input, details::checked_arithmetic_impl<T>::MIN)
        || cmp_less(details::checked_arithmetic_impl<T>::MAX, input)) {
        throw_overflow_exception();
    }

    return R(input);
}

template<typename R, typename T>
R checked_cast(const T& input, const R& length) {
    if (cmp_less(input, details::checked_arithmetic_impl<T>::MIN) || !cmp_less(input, length)) {
        throw_overflow_exception();
    }

    return R(input);
}

}  // namespace kmm
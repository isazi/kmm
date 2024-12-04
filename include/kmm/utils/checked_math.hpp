#pragma once

#include <limits>
#include <type_traits>

namespace kmm {

namespace detail {

template<typename L, typename R, typename T>
struct checked_arithmetic_impl;

#define KMM_IMPL_CHECKED_ARITHMETIC(T, ADD_FUN, SUB_FUN, MUL_FUN) \
    template<>                                                    \
    struct checked_arithmetic_impl<T, T, T> {                     \
        static bool add(T lhs, T rhs, T* result) {                \
            return ADD_FUN(lhs, rhs, result) == false;            \
        }                                                         \
                                                                  \
        static bool sub(T lhs, T rhs, T* result) {                \
            return SUB_FUN(lhs, rhs, result) == false;            \
        }                                                         \
                                                                  \
        static bool mul(T lhs, T rhs, T* result) {                \
            return MUL_FUN(lhs, rhs, result) == false;            \
        }                                                         \
    };

KMM_IMPL_CHECKED_ARITHMETIC(
    signed int,
    __builtin_sadd_overflow,
    __builtin_ssub_overflow,
    __builtin_smul_overflow
)

KMM_IMPL_CHECKED_ARITHMETIC(
    signed long,
    __builtin_saddl_overflow,
    __builtin_ssubl_overflow,
    __builtin_smull_overflow
)

KMM_IMPL_CHECKED_ARITHMETIC(
    signed long long,
    __builtin_saddll_overflow,
    __builtin_ssubll_overflow,
    __builtin_smulll_overflow
)

KMM_IMPL_CHECKED_ARITHMETIC(
    unsigned int,
    __builtin_uadd_overflow,
    __builtin_usub_overflow,
    __builtin_umul_overflow
)

KMM_IMPL_CHECKED_ARITHMETIC(
    unsigned long,
    __builtin_uaddl_overflow,
    __builtin_usubl_overflow,
    __builtin_umull_overflow
)

KMM_IMPL_CHECKED_ARITHMETIC(
    unsigned long long,
    __builtin_uaddll_overflow,
    __builtin_usubll_overflow,
    __builtin_umulll_overflow
)

#define KMM_IMPL_CHECKED_ARITHMETIC_FORWARD(T, R)                         \
    template<>                                                            \
    struct checked_arithmetic_impl<T, T, T> {                             \
        static bool add(T lhs, T rhs, T* result) {                        \
            R temp = static_cast<R>(lhs) + static_cast<R>(rhs);           \
            *result = static_cast<T>(temp);                               \
            return temp >= static_cast<R>(std::numeric_limits<T>::min())  \
                && temp <= static_cast<R>(std::numeric_limits<T>::max()); \
        }                                                                 \
                                                                          \
        static bool sub(T lhs, T rhs, T* result) {                        \
            R temp = static_cast<R>(lhs) - static_cast<R>(rhs);           \
            *result = static_cast<T>(temp);                               \
            return temp >= static_cast<R>(std::numeric_limits<T>::min())  \
                && temp <= static_cast<R>(std::numeric_limits<T>::max()); \
        }                                                                 \
                                                                          \
        static bool mul(T lhs, T rhs, T* result) {                        \
            R temp = static_cast<R>(lhs) * static_cast<R>(rhs);           \
            *result = static_cast<T>(temp);                               \
            return temp >= static_cast<R>(std::numeric_limits<T>::min())  \
                && temp <= static_cast<R>(std::numeric_limits<T>::max()); \
        }                                                                 \
    };

KMM_IMPL_CHECKED_ARITHMETIC_FORWARD(signed short, signed int)
KMM_IMPL_CHECKED_ARITHMETIC_FORWARD(unsigned short, signed int)

KMM_IMPL_CHECKED_ARITHMETIC_FORWARD(signed char, signed int)
KMM_IMPL_CHECKED_ARITHMETIC_FORWARD(unsigned char, signed int)
KMM_IMPL_CHECKED_ARITHMETIC_FORWARD(char, signed int)

template<typename L, typename R, typename = void>
struct checked_compare_impl {
    static bool is_less(const L& left, const R& right) {
        return left < right;
    }

    static bool is_equal(const L& left, const R& right) {
        return left == right;
    }
};

template<typename L, typename R>
struct checked_compare_impl<L, R, std::enable_if_t<!std::is_signed_v<L> && std::is_signed_v<R>>> {
    using UR = std::make_unsigned_t<R>;

    static bool is_less(const L& left, const R& right) {
        return right >= static_cast<R>(0)
            && checked_compare_impl<L, UR>::is_less(left, static_cast<UR>(right));
    }

    static bool is_equal(const L& left, const R& right) {
        return right >= static_cast<R>(0)
            && checked_compare_impl<L, UR>::is_equal(left, static_cast<UR>(right));
    }
};

template<typename L, typename R>
struct checked_compare_impl<L, R, std::enable_if_t<std::is_signed_v<L> && !std::is_signed_v<R>>> {
    using UL = std::make_unsigned_t<L>;

    static bool is_less(const L& left, const R& right) {
        return left < static_cast<L>(0)
            || checked_compare_impl<UL, R>::is_less(static_cast<UL>(left), right);
    }

    static bool is_equal(const L& left, const R& right) {
        return left >= static_cast<L>(0)
            && checked_compare_impl<UL, R>::is_equal(static_cast<UL>(left), right);
    }
};

};  // namespace detail

[[noreturn]] void throw_overflow_exception();

/**
 *  Performs checked addition of two values, throwing an exception on overflow.
 */
template<typename T>
T checked_add(T left, T right) {
    T result;

    if (!detail::checked_arithmetic_impl<T, T, T>::add(left, right, &result)) {
        throw_overflow_exception();
    }

    return result;
}

/**
 * Performs checked subtraction of two values, throwing an exception on overflow.
 */
template<typename T>
T checked_sub(T left, T right) {
    T result;

    if (!detail::checked_arithmetic_impl<T, T, T>::sub(left, right, &result)) {
        throw_overflow_exception();
    }

    return result;
}

/**
 * Performs checked multiplication of two values, throwing an exception on overflow.
 */
template<typename T>
T checked_mul(T left, T right) {
    T result;

    if (!detail::checked_arithmetic_impl<T, T, T>::mul(left, right, &result)) {
        throw_overflow_exception();
    }

    return result;
}

/**
 * Performs checked division of two values, throwing an exception on division by zero.
 */
template<typename T>
T checked_div(T left, T right) {
    if (right == T {0}) {
        throw_overflow_exception();
    }

    return left / right;
}

/**
 * Negates the given input, throwing an exception if the result overflows.
 */
template<typename T>
T checked_negate(T value) {
    return checked_sub(static_cast<T>(0), value);
}

/**
 *  Performs checked addition of two values, clamping to the minimum/maximum value on overflow.
 */
template<typename T>
T saturating_add(T left, T right) {
    T result;

    if (!detail::checked_arithmetic_impl<T, T, T>::add(left, right, &result)) {
        return right < 0 ? std::numeric_limits<T>::min() : std::numeric_limits<T>::max();
    }

    return result;
}

/**
 * Performs checked subtraction of two values, clamping to the minimum/maximum value on overflow.
 */
template<typename T>
T saturating_sub(T left, T right) {
    T result;

    if (!detail::checked_arithmetic_impl<T, T, T>::sub(left, right, &result)) {
        return right >= 0 ? std::numeric_limits<T>::min() : std::numeric_limits<T>::max();
    }

    return result;
}

/**
 * Performs checked multiplication of two values, clamping to the minimum/maximum value on overflow.
 */
template<typename T>
T saturating_mul(T left, T right) {
    T result;

    if (!detail::checked_arithmetic_impl<T, T, T>::mul(left, right, &result)) {
        return (left >= 0) == (right >= 0) ? std::numeric_limits<T>::max()
                                           : std::numeric_limits<T>::min();
    }

    return result;
}

/**
 * Negates the given input, clamping to the minimum/maximum value on overflow.
 */
template<typename T>
T saturating_negate(T value) {
    T result;

    if (!detail::checked_arithmetic_impl<T, T, T>::sub(T(0), value, &result)) {
        return value >= 0 ? std::numeric_limits<T>::min() : std::numeric_limits<T>::max();
    }

    return result;
}

/**
 * Returns true if `left < right`. This function correctly handles different operand types.
 */
template<typename L, typename R>
bool compare_less(const L& left, const R& right) {
    return detail::checked_compare_impl<L, R>::is_less(left, right);
}

/**
 * Returns true if `left > right`. This function correctly handles different operand types.
 */
template<typename L, typename R>
bool compare_greater(const L& left, const R& right) {
    return compare_less(right, left);
}

/**
 * Returns true if `left == right`. This function correctly handles different operand types.
 */
template<typename L, typename R>
bool compare_equal(const L& left, const R& right) {
    return detail::checked_compare_impl<L, R>::is_equal(left, right);
}

/**
 * Returns `true` if the given value of integral type `T` can safely be cast to another integral
 * type `U`, and `false` otherwise.
 */
template<typename U, typename T>
bool in_range(const T& value) {
    return !compare_less(value, std::numeric_limits<U>::lowest())
        && !compare_greater(value, std::numeric_limits<U>::max());
}

/**
 * Returns `true` if the given value of integral type `T` is in the range `0` to `length` (not
 * inclusive). Useful for checking if an value can be used to index into an array of size `length`.
 */
template<typename U, typename T>
constexpr bool in_range(const T& value, const U& length) {
    return !compare_less(value, static_cast<T>(0)) && compare_less(value, length);
}

/**
 * Performs a checked cast of a value of type `T` to a different type `U`, throwing an exception if
 * the value is out of range for type `U`.
 */
template<typename U, typename T>
constexpr U checked_cast(const T& value) {
    if (!in_range<U>(value)) {
        throw_overflow_exception();
    }

    return static_cast<U>(value);
}

/**
 * Performs a checked cast of a value of type `T` to a different type `U`, throwing an exception if
 * the value is not in the range `0` to `length` (not inclusive).
 */
template<typename U, typename T>
constexpr U checked_cast(const T& value, const U& length) {
    if (compare_less(value, static_cast<T>(0)) || !compare_less(value, length)) {
        throw_overflow_exception();
    }

    return checked_cast<U>(value);
}

/**
 * Computes the sum of an array of values, throwing an exception on overflow.
 */
template<typename U, typename T>
U checked_sum(const T* begin, const T* end) {
    if (begin == end) {
        return U {};
    }

    bool is_valid = in_range<U>(*begin);
    U result = static_cast<U>(*begin);

    for (const T* it = begin + 1; it != end; it++) {
        is_valid &= detail::checked_arithmetic_impl<U, T, U>::add(result, *it, &result);
    }

    if (!is_valid) {
        throw_overflow_exception();
    }

    return result;
}

template<typename T>
T checked_sum(const T* begin, const T* end) {
    return checked_sum<T, T>(begin, end);
}

/**
 * Computes the product of an array of values, throwing an exception on overflow.
 */
template<typename U, typename T>
U checked_product(const T* begin, const T* end) {
    if (begin == end) {
        return static_cast<U>(1);
    }

    bool is_valid = in_range<U>(*begin);
    U result = static_cast<U>(*begin);

    for (const T* it = begin + 1; it != end; it++) {
        is_valid &= detail::checked_arithmetic_impl<U, T, U>::mul(result, *it, &result);
    }

    if (!is_valid) {
        throw_overflow_exception();
    }

    return result;
}

template<typename T>
T checked_product(const T* begin, const T* end) {
    return checked_product<T, T>(begin, end);
}

}  // namespace kmm
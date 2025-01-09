#pragma once

#include "kmm/core/reduction.hpp"
#include "kmm/core/macros.hpp"

namespace kmm {

template<typename T, Reduction Op, typename = void>
struct ReductionOperator;

template<typename T>
struct ReductionOperator<
    T,
    Reduction::Sum,
    std::void_t<decltype(std::declval<T>() + std::declval<T>())>> {
    static KMM_HOST_DEVICE T identity() {
        return T(0);
    }

    KMM_HOST_DEVICE T operator()(T a, T b) {
        return static_cast<T>(a + b);
    }
};

template<typename T>
struct ReductionOperator<
    T,
    Reduction::Product,
    std::void_t<decltype(std::declval<T>() * std::declval<T>())>> {
    static KMM_HOST_DEVICE T identity() {
        return T(1);
    }

    KMM_HOST_DEVICE T operator()(T a, T b) {
        return static_cast<T>(a * b);
    }
};

template<typename T>
struct ReductionOperator<T, Reduction::Min, std::enable_if_t<std::numeric_limits<T>::is_specialized>> {
    static constexpr T MAX_VALUE = std::numeric_limits<T>::max();

    static KMM_HOST_DEVICE T identity() {
        return MAX_VALUE;
    }

    KMM_HOST_DEVICE T operator()(T a, T b) {
        return a < b ? a : b;
    }
};

template<typename T>
struct ReductionOperator<T, Reduction::Max, std::enable_if_t<std::numeric_limits<T>::is_specialized>> {
    static constexpr T MIN_VALUE = std::numeric_limits<T>::lowest();

    static KMM_HOST_DEVICE T identity() {
        return MIN_VALUE;
    }

    KMM_HOST_DEVICE T operator()(T a, T b) {
        return b < a ? a : b;
    }
};

template<typename T>
struct ReductionOperator<T, Reduction::BitOr, std::enable_if_t<std::is_integral_v<T>>> {
    static KMM_HOST_DEVICE T identity() {
        return T(0);
    }

    KMM_HOST_DEVICE T operator()(T a, T b) {
        return static_cast<T>(a | b);
    }
};

template<typename T>
struct ReductionOperator<T, Reduction::BitAnd, std::enable_if_t<std::is_integral_v<T>>> {
    static KMM_HOST_DEVICE T identity() {
        // Note: we need the static cast here since decltype(~(short)0) == int
        return static_cast<T>(~T(0));
    }

    KMM_HOST_DEVICE T operator()(T a, T b) {
        return static_cast<T>(a & b);
    }
};

template<>
struct ReductionOperator<float, Reduction::Min> {
    static KMM_HOST_DEVICE float identity() {
        return INFINITY;
    }

    KMM_HOST_DEVICE float operator()(float a, float b) {
        return fminf(a, b);
    }
};

template<>
struct ReductionOperator<float, Reduction::Max> {
    static KMM_HOST_DEVICE float identity() {
        return -INFINITY;
    }

    KMM_HOST_DEVICE float operator()(float a, float b) {
        return fmaxf(a, b);
    }
};

template<>
struct ReductionOperator<double, Reduction::Max> {
    static KMM_HOST_DEVICE double identity() {
        return -double(INFINITY);
    }

    KMM_HOST_DEVICE double operator()(double a, double b) {
        return fmax(a, b);
    }
};

template<>
struct ReductionOperator<double, Reduction::Min> {
    static KMM_HOST_DEVICE double identity() {
        return double(INFINITY);
    }

    KMM_HOST_DEVICE double operator()(double a, double b) {
        return fmin(a, b);
    }
};

template<typename T, Reduction Op, typename = void>
struct IsReductionSupported: std::false_type {};

template<typename T, Reduction Op>
struct IsReductionSupported<T, Op, std::void_t<decltype(ReductionOperator<T, Op>())>>:
    std::true_type {};

}  // namespace kmm
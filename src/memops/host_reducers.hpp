#pragma once

#include "kmm/core/reduction.hpp"
#include "kmm/utils/macros.hpp"

namespace kmm {

template<typename T, ReductionOp Op, typename = void>
struct ReductionFunctor;

template<typename T>
struct ReductionFunctor<T, ReductionOp::Sum, std::void_t<decltype(std::declval<T>() + std::declval<T>())>> {
    static KMM_HOST_DEVICE T identity() {
        return T(0);
    }

    static KMM_HOST_DEVICE T combine(T a, T b) {
        return static_cast<T>(a + b);
    }
};

template<typename T>
struct ReductionFunctor<T, ReductionOp::Product, std::void_t<decltype(std::declval<T>() * std::declval<T>())>> {
    static KMM_HOST_DEVICE T identity() {
        return T(1);
    }

    static KMM_HOST_DEVICE T combine(T a, T b) {
        return static_cast<T>(a * b);
    }
};

template<typename T>
struct ReductionFunctor<T, ReductionOp::Min> {
    static constexpr T MAX_VALUE = std::numeric_limits<T>::max();

    static KMM_HOST_DEVICE T identity() {
        return MAX_VALUE;
    }

    static KMM_HOST_DEVICE T combine(T a, T b) {
        return a < b ? a : b;
    }
};

template<typename T>
struct ReductionFunctor<T, ReductionOp::Max> {
    static constexpr T MIN_VALUE = std::numeric_limits<T>::min();
    ;

    static KMM_HOST_DEVICE T identity() {
        return MIN_VALUE;
    }

    static KMM_HOST_DEVICE T combine(T a, T b) {
        return b < a ? a : b;
    }
};

template<typename T>
struct ReductionFunctor<T, ReductionOp::BitOr, std::enable_if_t<std::is_integral_v<T>>> {
    static KMM_HOST_DEVICE T identity() {
        return T(0);
    }

    static KMM_HOST_DEVICE T combine(T a, T b) {
        return static_cast<T>(a | b);
    }
};

template<typename T>
struct ReductionFunctor<T, ReductionOp::BitAnd, std::enable_if_t<std::is_integral_v<T>>> {
    static KMM_HOST_DEVICE T identity() {
        return ~T(0);
    }

    static KMM_HOST_DEVICE T combine(T a, T b) {
        return static_cast<T>(a & b);
    }
};

template<>
struct ReductionFunctor<float, ReductionOp::Min> {
    static KMM_HOST_DEVICE float identity() {
        return INFINITY;
    }

    static KMM_HOST_DEVICE float combine(float a, float b) {
        return fminf(a, b);
    }
};

template<>
struct ReductionFunctor<float, ReductionOp::Max> {
    static KMM_HOST_DEVICE float identity() {
        return -INFINITY;
    }

    static KMM_HOST_DEVICE float combine(float a, float b) {
        return fmaxf(a, b);
    }
};

template<>
struct ReductionFunctor<double, ReductionOp::Max> {
    static KMM_HOST_DEVICE double identity() {
        return -double(INFINITY);
    }

    static KMM_HOST_DEVICE double combine(double a, double b) {
        return fmax(a, b);
    }
};

template<>
struct ReductionFunctor<double, ReductionOp::Min> {
    static KMM_HOST_DEVICE double identity() {
        return double(INFINITY);
    }

    static KMM_HOST_DEVICE double combine(double a, double b) {
        return fmin(a, b);
    }
};

template<typename T, ReductionOp Op, typename = void>
struct ReductionFunctorSupported: std::false_type {};

template<typename T, ReductionOp Op>
struct ReductionFunctorSupported<T, Op, std::void_t<decltype(ReductionFunctor<T, Op>())>>:
    std::true_type {};

}
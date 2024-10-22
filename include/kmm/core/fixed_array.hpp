#pragma once

#include "kmm/utils/macros.hpp"

namespace kmm {

KMM_HOST_DEVICE bool is_less(size_t a, size_t b) {
    return a < b;
}

template<typename T, size_t N>
struct fixed_array {
    KMM_HOST_DEVICE
    T& operator[](size_t axis) {
        if (axis >= N) {
            __builtin_unreachable();
        }

        return item[axis];
    }

    KMM_HOST_DEVICE
    const T& operator[](size_t axis) const {
        if (axis >= N) {
            __builtin_unreachable();
        }

        return item[axis];
    }

    T item[N] = {};
};

template<typename T>
struct fixed_array<T, 0> {
    KMM_HOST_DEVICE
    T& operator[](size_t axis) {
        __builtin_unreachable();
    }

    KMM_HOST_DEVICE
    const T& operator[](size_t axis) const {
        __builtin_unreachable();
    }
};

template<typename T>
struct fixed_array<T, 1> {
    KMM_HOST_DEVICE
    T& operator[](size_t axis) {
        return x;
    }

    KMM_HOST_DEVICE
    const T& operator[](size_t axis) const {
        return x;
    }

    KMM_HOST_DEVICE
    operator T() const {
        return x;
    }

    KMM_HOST_DEVICE
    fixed_array& operator=(T value) {
        x = value;
        return *this;
    }

    T x {};
};

template<typename T>
struct fixed_array<T, 2> {
    KMM_HOST_DEVICE
    T& operator[](size_t axis) {
        switch (axis) {
            case 0:
                return x;
            case 1:
                return y;
            default:
                __builtin_unreachable();
        }
    }

    KMM_HOST_DEVICE
    const T& operator[](size_t axis) const {
        switch (axis) {
            case 0:
                return x;
            case 1:
                return y;
            default:
                __builtin_unreachable();
        }
    }

    T x {};
    T y {};
};

template<typename T>
struct fixed_array<T, 3> {
    KMM_HOST_DEVICE
    T& operator[](size_t axis) {
        switch (axis) {
            case 0:
                return x;
            case 1:
                return y;
            case 2:
                return z;
            default:
                __builtin_unreachable();
        }
    }

    KMM_HOST_DEVICE
    const T& operator[](size_t axis) const {
        switch (axis) {
            case 0:
                return x;
            case 1:
                return y;
            case 2:
                return z;
            default:
                __builtin_unreachable();
        }
    }

    T x {};
    T y {};
    T z {};
};

template<typename T>
struct fixed_array<T, 4> {
    KMM_HOST_DEVICE
    T& operator[](size_t axis) {
        switch (axis) {
            case 0:
                return x;
            case 1:
                return y;
            case 2:
                return z;
            case 3:
                return w;
            default:
                __builtin_unreachable();
        }
    }

    KMM_HOST_DEVICE
    const T& operator[](size_t axis) const {
        switch (axis) {
            case 0:
                return x;
            case 1:
                return y;
            case 2:
                return z;
            case 3:
                return w;
            default:
                __builtin_unreachable();
        }
    }

    T x {};
    T y {};
    T z {};
    T w {};
};

template<typename T, size_t N, typename U, size_t M>
KMM_HOST_DEVICE bool operator==(const fixed_array<T, N>& lhs, const fixed_array<U, M>& rhs) {
    if (N != M) {
        return false;
    }

    bool result = true;

    for (size_t i = 0; is_less(i, N); i++) {
        result &= lhs[i] == rhs[i];
    }

    return result;
}

template<typename T, size_t N, typename U, size_t M>
KMM_HOST_DEVICE bool operator!=(const fixed_array<T, N>& lhs, const fixed_array<U, M>& rhs) {
    return !(lhs == rhs);
}

}  // namespace kmm

#include <iostream>

#include "fmt/ostream.h"

namespace kmm {

template<typename T, size_t N>
std::ostream& operator<<(std::ostream& stream, const fixed_array<T, N>& p) {
    stream << "{";
    for (size_t i = 0; is_less(i, N); i++) {
        if (i != 0) {
            stream << ", ";
        }

        stream << p[i];
    }

    return stream << "}";
}
}  // namespace kmm

template<typename T, size_t N>
struct fmt::formatter<kmm::fixed_array<T, N>>: fmt::ostream_formatter {};

#include "kmm/utils/hash_utils.hpp"

template<size_t N, typename T>
struct std::hash<kmm::fixed_array<T, N>> {
    size_t operator()(const kmm::fixed_array<T, N>& p) const {
        size_t result = 0;
        for (size_t i = 0; kmm::is_less(i, N); i++) {
            kmm::hash_combine(result, p[i]);
        }
        return result;
    }
};

template<typename T>
struct std::hash<kmm::fixed_array<T, 0>> {
    size_t operator()(const kmm::fixed_array<T, 0>& p) const {
        return 0;
    }
};
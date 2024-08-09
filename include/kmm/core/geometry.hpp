#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "kmm/utils/macros.hpp"

namespace kmm {

static constexpr size_t MAX_DIMS = 4;
using default_geometry_type = int64_t;

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

template<size_t N, typename T = default_geometry_type>
class point: public fixed_array<T, N> {
  public:
    using storage_type = fixed_array<T, N>;

    KMM_HOST_DEVICE
    explicit constexpr point(const storage_type& storage) : storage_type(storage) {}

    KMM_HOST_DEVICE
    constexpr point() {
        for (size_t i = 0; i < N; i++) {
            (*this)[i] = T {};
        }
    }

    template<typename... Ts, typename = typename std::enable_if<(sizeof...(Ts) + 1 == N)>::type>
    KMM_HOST_DEVICE point(T first, Ts&&... args) :
        storage_type {static_cast<T>(first), static_cast<T>(args)...} {}

    template<size_t M, typename U>
    KMM_HOST_DEVICE static constexpr point from(const fixed_array<U, M>& that) {
        point result;

        for (size_t i = 0; i < N && i < M; i++) {
            result[i] = that[i];
        }

        return result;
    }

    KMM_HOST_DEVICE
    static constexpr point fill(T value) {
        point result;

        for (size_t i = 0; i < N; i++) {
            result[i] = value;
        }

        return result;
    }

    KMM_HOST_DEVICE
    static constexpr point zero() {
        return fill(static_cast<T>(0));
    }

    KMM_HOST_DEVICE
    static constexpr point one() {
        return fill(static_cast<T>(1));
    }

    KMM_HOST_DEVICE
    T get(size_t axis) const {
        return axis < N ? (*this)[axis] : static_cast<T>(0);
    }

    template<typename F, typename U = std::invoke_result_t<F, T>>
    KMM_HOST_DEVICE point<N, U> map(F fun) const {
        point<N, U> result;

        for (size_t i = 0; i < N; i++) {
            result[i] = fun((*this)[i]);
        }

        return result;
    }

    KMM_HOST_DEVICE
    bool equals(const point& that) const {
        bool is_equal = true;

        for (size_t i = 0; i < N; i++) {
            is_equal &= (*this)[i] == that[i];
        }

        return is_equal;
    }
};

template<size_t N, typename T = default_geometry_type>
class dim: public fixed_array<T, N> {
  public:
    using storage_type = fixed_array<T, N>;

    KMM_HOST_DEVICE
    explicit constexpr dim(const storage_type& storage) : storage_type(storage) {}

    KMM_HOST_DEVICE
    constexpr dim() {
        for (size_t i = 0; i < N; i++) {
            (*this)[i] = T {1};
        }
    }

    template<typename... Ts, typename = typename std::enable_if<(sizeof...(Ts) + 1 == N)>::type>
    KMM_HOST_DEVICE dim(T first, Ts&&... args) :
        storage_type {static_cast<T>(first), static_cast<T>(args)...} {}

    KMM_HOST_DEVICE
    static constexpr dim from_point(const point<N, T>& that) {
        return dim {that};
    }

    template<size_t M, typename U>
    KMM_HOST_DEVICE static constexpr dim from(const fixed_array<U, M>& that) {
        return from_point(point<N, T>::from(that));
    }

    KMM_HOST_DEVICE
    static constexpr dim zero() {
        return from_point(point<N, T>::zero());
    }

    KMM_HOST_DEVICE
    static constexpr dim one() {
        return from_point(point<N, T>::one());
    }

    KMM_HOST_DEVICE
    point<N, T> to_point() const {
        return point<N, T>::from(*this);
    }

    KMM_HOST_DEVICE
    bool is_empty() const {
        bool is_empty = false;

        for (size_t i = 0; i < N; i++) {
            is_empty |= (*this)[i] <= static_cast<T>(0);
        }

        return is_empty;
    }

    KMM_HOST_DEVICE
    T get(size_t i) const {
        return i < N ? (*this)[i] : static_cast<T>(1);
    }

    KMM_HOST_DEVICE
    T volume() const {
        if (is_empty()) {
            return static_cast<T>(0);
        }

        if (N == 0) {
            return static_cast<T>(1);
        }

        T volume = (*this)[0];

        for (size_t i = 1; i < N; i++) {
            volume *= (*this)[i];
        }

        return volume;
    }

    KMM_HOST_DEVICE
    dim intersection(const dim& that) const {
        dim<N, T> new_sizes;

        for (size_t i = 0; i < N; i++) {
            if (that[i] <= 0 || (*this)[i] <= 0) {
                new_sizes[i] = static_cast<T>(0);
            } else if ((*this)[i] <= that[i]) {
                new_sizes[i] = (*this)[i];
            } else {
                new_sizes[i] = that[i];
            }
        }

        return {new_sizes};
    }

    KMM_HOST_DEVICE
    bool overlaps(const dim& that) const {
        return !this->is_empty() && !that.is_empty();
    }

    KMM_HOST_DEVICE
    bool contains(const dim& that) const {
        return that.is_empty() || intersection(that) == that;
    }

    KMM_HOST_DEVICE
    bool contains(const point<N, T>& that) const {
        for (size_t i = 0; i < N; i++) {
            if (that[i] < static_cast<T>(0) || that[i] >= (*this)[i]) {
                return false;
            }
        }

        return true;
    }
};

template<size_t N, typename T = default_geometry_type>
class rect {
  public:
    point<N, T> offset;
    dim<N, T> sizes;

    KMM_HOST_DEVICE
    rect(point<N, T> offset, dim<N, T> sizes) : offset(offset), sizes(sizes) {}

    KMM_HOST_DEVICE
    rect(dim<N, T> sizes) : rect(point<N, T>::zero(), sizes) {}

    KMM_HOST_DEVICE
    rect() : rect(dim<N, T>::zero()) {}

    template<size_t M, typename U>
    KMM_HOST_DEVICE static constexpr rect from(const rect<M, U>& that) {
        return {point<N, T>::from(that.offset()), dim<N, T>::from(that.sizes())};
    }

    KMM_HOST_DEVICE
    T size(size_t axis) const {
        return this->sizes.get(axis);
    }

    KMM_HOST_DEVICE
    T begin(size_t axis) const {
        return this->offset.get(axis);
    }

    KMM_HOST_DEVICE
    T end(size_t axis) const {
        return begin(axis) + (size(axis) < 0 ? 0 : size(axis));
    }

    KMM_HOST_DEVICE
    point<N, T> begin() const {
        return this->offset;
    }

    KMM_HOST_DEVICE
    point<N, T> end() const {
        point<N, T> result;
        for (size_t i = 0; i < N; i++) {
            result[i] = end(i);
        }

        return result;
    }

    KMM_HOST_DEVICE
    T size() const {
        return this->sizes.volume();
    }

    KMM_HOST_DEVICE
    bool is_empty() const {
        return this->sizes.is_empty();
    }

    KMM_HOST_DEVICE
    rect intersection(const rect& that) const {
        point<N, T> new_offset;
        dim<N, T> new_sizes;

        for (size_t i = 0; i < N; i++) {
            if (this->offset[i] < that.offset[i]) {
                new_offset[i] = that.offset[i];

                if (this->sizes[i] <= that.offset[i] - this->offset[i]) {
                    return rect {};
                }

                new_sizes[i] = this->sizes[i] + this->offset[i] - that.offset[i];

                if (that.sizes[i] < new_sizes[i]) {
                    new_sizes[i] = that.sizes[i];
                }
            } else {
                new_offset[i] = this->offset[i];

                if (that.sizes[i] <= this->offset[i] - that.offset[i]) {
                    return rect {};
                }

                new_sizes[i] = that.sizes[i] + that.offset[i] - this->offset[i];

                if (this->sizes[i] < new_sizes[i]) {
                    new_sizes[i] = this->sizes[i];
                }
            }
        }

        return {new_offset, new_sizes};
    }

    KMM_HOST_DEVICE
    bool overlaps(const rect& that) const {
        return !intersection(that).is_empty();
    }

    KMM_HOST_DEVICE
    bool contains(const rect& that) const {
        return that.is_empty() || intersection(that) == that;
    }

    KMM_HOST_DEVICE
    bool contains(const point<N, T>& that) const {
        return contains(rect {that, dim<N, T>::one()});
    }

    KMM_HOST_DEVICE
    rect intersection(const dim<N, T>& that) const {
        return intersection(rect<N, T> {that});
    }

    KMM_HOST_DEVICE
    bool overlaps(const dim<N, T>& that) const {
        return overlaps(rect<N, T> {that});
    }

    KMM_HOST_DEVICE
    bool contains(const dim<N, T>& that) const {
        return contains(rect<N, T> {that});
    }
};

template<typename... Ts>
point(Ts...) -> point<sizeof...(Ts)>;

template<typename... Ts>
dim(Ts...) -> dim<sizeof...(Ts)>;

template<size_t N, typename T>
rect(point<N, T> offset, dim<N, T> sizes) -> rect<N, T>;

template<size_t N, typename T>
rect(dim<N, T> sizes) -> rect<N, T>;

template<size_t N, typename T>
KMM_HOST_DEVICE bool operator==(const point<N, T>& a, const point<N, T>& b) {
    return a.equals(b);
}

template<size_t N, typename T>
KMM_HOST_DEVICE bool operator!=(const point<N, T>& a, const point<N, T>& b) {
    return !(a == b);
}

template<size_t N, typename T>
KMM_HOST_DEVICE bool operator==(const dim<N, T>& a, const dim<N, T>& b) {
    return a.to_point() == b.to_point();
}

template<size_t N, typename T>
KMM_HOST_DEVICE bool operator!=(const dim<N, T>& a, const dim<N, T>& b) {
    return !(a == b);
}

template<size_t N, typename T>
KMM_HOST_DEVICE bool operator==(const rect<N, T>& a, const rect<N, T>& b) {
    return a.offset == b.offset && a.sizes == b.sizes;
}

template<size_t N, typename T>
KMM_HOST_DEVICE bool operator!=(const rect<N, T>& a, const rect<N, T>& b) {
    return !(a == b);
}

#define KMM_POINT_OPERATOR_IMPL(OP)                                                       \
    template<size_t N, typename T>                                                        \
    KMM_HOST_DEVICE point<N, T> operator OP(const point<N, T>& a, const point<N, T>& b) { \
        point<N, T> result;                                                               \
        for (size_t i = 0; i < N; i++) {                                                  \
            result[i] = a[i] OP b[i];                                                     \
        }                                                                                 \
                                                                                          \
        return result;                                                                    \
    }

KMM_POINT_OPERATOR_IMPL(+)
KMM_POINT_OPERATOR_IMPL(-)
KMM_POINT_OPERATOR_IMPL(*)
KMM_POINT_OPERATOR_IMPL(/)

}  // namespace kmm

#include <iostream>
namespace kmm {

template<size_t N, typename T>
std::ostream& operator<<(std::ostream& stream, const point<N, T>& p) {
    stream << "{";
    for (size_t i = 0; i < N; i++) {
        if (i != 0) {
            stream << ", ";
        }

        stream << p[i];
    }

    return stream << "}";
}

template<size_t N, typename T>
std::ostream& operator<<(std::ostream& stream, const dim<N, T>& p) {
    return stream << p.to_point();
}

template<size_t N, typename T>
std::ostream& operator<<(std::ostream& stream, const rect<N, T>& p) {
    return stream << p.offset << "..." << (p.offset + p.sizes.to_point());
}
}  // namespace kmm

#include "fmt/ostream.h"

template<size_t N, typename T>
struct fmt::formatter<kmm::point<N, T>>: fmt::ostream_formatter {};

template<size_t N, typename T>
struct fmt::formatter<kmm::dim<N, T>>: fmt::ostream_formatter {};

template<size_t N, typename T>
struct fmt::formatter<kmm::rect<N, T>>: fmt::ostream_formatter {};
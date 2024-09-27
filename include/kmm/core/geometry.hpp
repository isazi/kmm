#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "fixed_array.hpp"

#include "kmm/utils/macros.hpp"

namespace kmm {

using default_geometry_type = int64_t;

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

    template<typename... Ts, typename = typename std::enable_if<(sizeof...(Ts) < N)>::type>
    KMM_HOST_DEVICE point(T first, Ts&&... args) {
        (*this)[0] = first;

        size_t index = 0;
        (((*this)[index++] = args), ...);
    }

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

    KMM_HOST_DEVICE
    T operator()(size_t axis = 0) const {
        return get(axis);
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

    template<typename... Ts, typename = typename std::enable_if<(sizeof...(Ts) < N)>::type>
    KMM_HOST_DEVICE dim(T first, Ts&&... args) {
        (*this)[0] = first;

        size_t index = 0;
        (((*this)[index++] = args), ...);
    }

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
        if constexpr (N == 0) {
            return static_cast<T>(1);
        }

        if (is_empty()) {
            return static_cast<T>(0);
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

    KMM_HOST_DEVICE
    T operator()(size_t axis = 0) const {
        return get(axis);
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

    template<typename... Ts, typename = typename std::enable_if<(sizeof...(Ts) < N)>::type>
    KMM_HOST_DEVICE rect(T first, Ts&&... args) :
        offset(point<N, T>::zero()),
        sizes(first, args...) {}

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
        bool is_empty = false;

        for (size_t i = 0; i < N; i++) {
            auto first_a = this->offset[i] < that.offset[i];

            auto ai = first_a ? this->offset[i] : that.offset[i];
            auto an = first_a ? this->sizes[i] : that.sizes[i];

            auto bi = !first_a ? this->offset[i] : that.offset[i];
            auto bn = !first_a ? this->sizes[i] : that.sizes[i];

            if (an <= bi - ai) {
                is_empty = true;
            }

            new_offset[i] = bi;
            new_sizes[i] = an - (bi - ai);

            if (bn < an - (bi - ai)) {
                new_sizes[i] = bn;
            }
        }

        if (is_empty) {
            return rect {};
        }

        return {new_offset, new_sizes};
    }

    KMM_HOST_DEVICE
    bool overlaps(const rect& that) const {
        bool overlapping = true;

        for (size_t i = 0; i < N; i++) {
            auto first_this = this->offset[i] < that.offset[i];

            auto ai = first_this ? this->offset[i] : that.offset[i];
            auto an = first_this ? this->sizes[i] : that.sizes[i];

            auto bi = !first_this ? this->offset[i] : that.offset[i];
            auto bn = !first_this ? this->sizes[i] : that.sizes[i];

            if (an <= bi - ai || bn <= 0) {
                overlapping = false;
            }
        }

        return overlapping;
    }

    KMM_HOST_DEVICE
    bool contains(const rect& that) const {
        if (that.is_empty()) {
            return true;
        }

        bool contain = true;

        for (size_t i = 0; i < N; i++) {
            auto ai = this->offset[i];
            auto an = this->sizes[i];

            auto bi = that.offset[i];
            auto bn = that.sizes[i];

            // We should have: `ai <= bi && ai+an >= bi+bn`.
            // The inverse is: `ai > bi || ai+an < bi+bn`
            // Rewritten we get these conditions:
            // * `ai > bi`, or
            // * `an <= bi - ai`, or
            // * `an - (bi - ai) < + bn`
            if (ai > bi) {
                contain = false;
            }

            if (an <= bi - ai) {
                contain = false;
            }

            if (an - (bi - ai) < bn) {
                contain = false;
            }
        }

        return contain;
    }

    KMM_HOST_DEVICE
    bool contains(const point<N, T>& that) const {
        bool contain = true;

        for (size_t i = 0; i < N; i++) {
            auto ai = this->offset[i];
            auto an = this->sizes[i];

            if (!(that[i] >= ai && that[i] - ai < an)) {
                contain = false;
            }
        }

        return contain;
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
KMM_HOST_DEVICE_NOINLINE point(Ts...)->point<sizeof...(Ts)>;

template<typename... Ts>
KMM_HOST_DEVICE_NOINLINE dim(Ts...)->dim<sizeof...(Ts)>;

template<size_t N, typename T>
KMM_HOST_DEVICE_NOINLINE rect(point<N, T> offset, dim<N, T> sizes)->rect<N, T>;

template<size_t N, typename T>
KMM_HOST_DEVICE_NOINLINE rect(dim<N, T> sizes)->rect<N, T>;

template<size_t N, typename T>
KMM_HOST_DEVICE bool operator==(const point<N, T>& a, const point<N, T>& b) {
    return (const fixed_array<T, N>&)a == (const fixed_array<T, N>&)b;
}

template<size_t N, typename T>
KMM_HOST_DEVICE bool operator!=(const point<N, T>& a, const point<N, T>& b) {
    return !(a == b);
}

template<size_t N, typename T>
KMM_HOST_DEVICE bool operator==(const dim<N, T>& a, const dim<N, T>& b) {
    return (const fixed_array<T, N>&)a == (const fixed_array<T, N>&)b;
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
    return stream << fixed_array<T, N>(p);
}

template<size_t N, typename T>
std::ostream& operator<<(std::ostream& stream, const dim<N, T>& p) {
    return stream << fixed_array<T, N>(p);
}

template<size_t N, typename T>
std::ostream& operator<<(std::ostream& stream, const rect<N, T>& p) {
    stream << "{";
    for (size_t i = 0; i < N; i++) {
        if (i != 0) {
            stream << ", ";
        }

        stream << p.offset[i] << "..." << (p.offset[i] + p.sizes[i]);
    }

    return stream << "}";
}
}  // namespace kmm

#include "fmt/ostream.h"

template<size_t N, typename T>
struct fmt::formatter<kmm::point<N, T>>: fmt::ostream_formatter {};

template<size_t N, typename T>
struct fmt::formatter<kmm::dim<N, T>>: fmt::ostream_formatter {};

template<size_t N, typename T>
struct fmt::formatter<kmm::rect<N, T>>: fmt::ostream_formatter {};

#include "kmm/utils/hash_utils.hpp"

template<size_t N, typename T>
struct std::hash<kmm::point<N, T>>: std::hash<kmm::fixed_array<T, N>> {};

template<size_t N, typename T>
struct std::hash<kmm::dim<N, T>>: std::hash<kmm::fixed_array<T, N>> {};

template<size_t N, typename T>
struct std::hash<kmm::rect<N, T>> {
    size_t operator()(const kmm::rect<N, T>& p) const {
        kmm::fixed_array<T, N> v[2] = {p.offset, p.sizes};
        return kmm::hash_range(v, v + 2);
    }
};

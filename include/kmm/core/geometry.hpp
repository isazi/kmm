#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "fixed_array.hpp"

#include "kmm/utils/macros.hpp"

namespace kmm {

using default_geometry_type = int64_t;

template<size_t N, typename T = default_geometry_type>
class Point: public fixed_array<T, N> {
  public:
    using storage_type = fixed_array<T, N>;

    KMM_HOST_DEVICE
    explicit constexpr Point(const storage_type& storage) : storage_type(storage) {}

    KMM_HOST_DEVICE
    constexpr Point() {
        for (size_t i = 0; i < N; i++) {
            (*this)[i] = T {};
        }
    }

    template<typename... Ts, typename = typename std::enable_if<(sizeof...(Ts) < N)>::type>
    KMM_HOST_DEVICE Point(T first, Ts&&... args) : Point() {
        (*this)[0] = first;

        size_t index = 0;
        (((*this)[++index] = args), ...);
    }

    template<size_t M, typename U>
    KMM_HOST_DEVICE static constexpr Point from(const fixed_array<U, M>& that) {
        Point result;

        for (size_t i = 0; i < N && is_less(i, M); i++) {
            result[i] = that[i];
        }

        return result;
    }

    KMM_HOST_DEVICE
    static constexpr Point fill(T value) {
        Point result;

        for (size_t i = 0; i < N; i++) {
            result[i] = value;
        }

        return result;
    }

    KMM_HOST_DEVICE
    static constexpr Point zero() {
        return fill(static_cast<T>(0));
    }

    KMM_HOST_DEVICE
    static constexpr Point one() {
        return fill(static_cast<T>(1));
    }

    KMM_HOST_DEVICE
    T get(size_t axis) const {
        return KMM_LIKELY(axis < N) ? (*this)[axis] : static_cast<T>(0);
    }

    KMM_HOST_DEVICE
    T operator()(size_t axis = 0) const {
        return get(axis);
    }

    template<size_t M>
    KMM_HOST_DEVICE Point<N + M> concat(const Point<M>& that) const {
        fixed_array<T, N + M> result;

        for (size_t i = 0; i < N; i++) {
            result[i] = (*this)[i];
        }

        for (size_t i = 0; is_less(i, M); i++) {
            result[N + i] = that[i];
        }

        return Point<N + M> {result};
    }
};

template<typename T>
class Point<0, T>: public fixed_array<T, 0> {
  public:
    using storage_type = fixed_array<T, 0>;

    KMM_HOST_DEVICE
    explicit constexpr Point(const storage_type& storage) : storage_type(storage) {}

    KMM_HOST_DEVICE
    constexpr Point() {}

    template<size_t M, typename U>
    KMM_HOST_DEVICE static constexpr Point from(const fixed_array<U, M>& that) {
        return {};
    }

    KMM_HOST_DEVICE
    static constexpr Point fill(T value) {
        return {};
    }

    KMM_HOST_DEVICE
    static constexpr Point zero() {
        return {};
    }

    KMM_HOST_DEVICE
    static constexpr Point one() {
        return {};
    }

    KMM_HOST_DEVICE
    T get(size_t axis) const {
        return T {};
    }

    KMM_HOST_DEVICE
    T operator()(size_t axis = 0) const {
        return get(axis);
    }

    template<size_t M>
    KMM_HOST_DEVICE Point<M> concat(const Point<M>& that) const {
        return that;
    }
};

template<size_t N, typename T = default_geometry_type>
class Dim: public fixed_array<T, N> {
  public:
    using storage_type = fixed_array<T, N>;

    KMM_HOST_DEVICE
    explicit constexpr Dim(const storage_type& storage) : storage_type(storage) {}

    KMM_HOST_DEVICE
    constexpr Dim() {
        for (size_t i = 0; i < N; i++) {
            (*this)[i] = static_cast<T>(1);
        }
    }

    template<typename... Ts, typename = typename std::enable_if<(sizeof...(Ts) < N)>::type>
    KMM_HOST_DEVICE Dim(T first, Ts&&... args) : Dim() {
        (*this)[0] = first;

        size_t index = 0;
        (((*this)[++index] = args), ...);
    }

    KMM_HOST_DEVICE
    static constexpr Dim from_point(const Point<N, T>& that) {
        return Dim {that};
    }

    template<size_t M, typename U>
    KMM_HOST_DEVICE static constexpr Dim from(const fixed_array<U, M>& that) {
        Dim result;

        for (size_t i = 0; i < N && is_less(i, M); i++) {
            result[i] = that[i];
        }

        return result;
    }

    KMM_HOST_DEVICE
    static constexpr Dim zero() {
        return from_point(Point<N, T>::zero());
    }

    KMM_HOST_DEVICE
    static constexpr Dim one() {
        return from_point(Point<N, T>::one());
    }

    KMM_HOST_DEVICE
    Point<N, T> to_point() const {
        return Point<N, T>::from(*this);
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
        return KMM_LIKELY(i < N) ? (*this)[i] : static_cast<T>(1);
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
    Dim intersection(const Dim& that) const {
        Dim<N, T> new_sizes;

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
    bool overlaps(const Dim& that) const {
        return !this->is_empty() && !that.is_empty();
    }

    KMM_HOST_DEVICE
    bool contains(const Dim& that) const {
        return that.is_empty() || intersection(that) == that;
    }

    KMM_HOST_DEVICE
    bool contains(const Point<N, T>& that) const {
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

    template<size_t M>
    KMM_HOST_DEVICE Dim<N + M> concat(const Dim<M>& that) const {
        return Dim<N + M>(to_point().concat(that.to_point()));
    }
};

template<typename T>
class Dim<0, T>: public fixed_array<T, 0> {
  public:
    using storage_type = fixed_array<T, 0>;

    KMM_HOST_DEVICE
    explicit constexpr Dim(const storage_type& storage) : storage_type(storage) {}

    KMM_HOST_DEVICE
    constexpr Dim() {}

    KMM_HOST_DEVICE
    static constexpr Dim from_point(const Point<0, T>& that) {
        return {};
    }

    template<size_t M, typename U>
    KMM_HOST_DEVICE static constexpr Dim from(const fixed_array<U, M>& that) {
        return that;
    }

    KMM_HOST_DEVICE
    static constexpr Dim zero() {
        return {};
    }

    KMM_HOST_DEVICE
    static constexpr Dim one() {
        return {};
    }

    KMM_HOST_DEVICE
    Point<0, T> to_point() const {
        return {};
    }

    KMM_HOST_DEVICE
    bool is_empty() const {
        return false;
    }

    KMM_HOST_DEVICE
    T get(size_t i) const {
        return static_cast<T>(1);
    }

    KMM_HOST_DEVICE
    T volume() const {
        return static_cast<T>(1);
    }

    KMM_HOST_DEVICE
    Dim intersection(const Dim& that) const {
        return {};
    }

    KMM_HOST_DEVICE
    bool overlaps(const Dim& that) const {
        return true;
    }

    KMM_HOST_DEVICE
    bool contains(const Dim& that) const {
        return true;
    }

    KMM_HOST_DEVICE
    bool contains(const Point<0, T>& that) const {
        return true;
    }

    KMM_HOST_DEVICE
    T operator()(size_t axis = 0) const {
        return get(axis);
    }

    template<size_t M>
    KMM_HOST_DEVICE Dim<M> concat(const Dim<M>& that) const {
        return that;
    }
};

template<size_t N, typename T = default_geometry_type>
class Range {
  public:
    Point<N, T> offset;
    Dim<N, T> sizes;

    KMM_HOST_DEVICE
    Range(Point<N, T> offset, Dim<N, T> sizes) : offset(offset), sizes(sizes) {}

    KMM_HOST_DEVICE
    Range(Dim<N, T> sizes) : Range(Point<N, T>::zero(), sizes) {}

    template<typename... Ts, typename = typename std::enable_if<(sizeof...(Ts) < N)>::type>
    KMM_HOST_DEVICE Range(T first, Ts&&... args) :
        offset(Point<N, T>::zero()),
        sizes(first, args...) {}

    KMM_HOST_DEVICE
    Range() : Range(Dim<N, T>::zero()) {}

    KMM_HOST_DEVICE static constexpr Range from_bounds(
        const Point<N, T>& begin,
        const Point<N, T>& end
    ) {
        return {begin, Dim<N, T>::from_point(end - begin)};
    }

    template<size_t M, typename U>
    KMM_HOST_DEVICE static constexpr Range from(const Range<M, U>& that) {
        return {Point<N, T>::from(that.offset()), Dim<N, T>::from(that.sizes())};
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
    Point<N, T> begin() const {
        return this->offset;
    }

    KMM_HOST_DEVICE
    Point<N, T> end() const {
        Point<N, T> result;
        for (size_t i = 0; is_less(i, N); i++) {
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
    Range intersection(const Range& that) const {
        Point<N, T> new_offset;
        Dim<N, T> new_sizes;
        bool is_empty = false;

        for (size_t i = 0; is_less(i, N); i++) {
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
            new_offset = Point<N, T>::zero();
            new_sizes = Dim<N, T>::zero();
        }

        return {new_offset, new_sizes};
    }

    KMM_HOST_DEVICE
    bool overlaps(const Range& that) const {
        bool overlapping = true;

        for (size_t i = 0; is_less(i, N); i++) {
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
    bool contains(const Range& that) const {
        if (that.is_empty()) {
            return true;
        }

        bool contain = true;

        for (size_t i = 0; is_less(i, N); i++) {
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
    bool contains(const Point<N, T>& that) const {
        bool contain = true;

        for (size_t i = 0; is_less(i, N); i++) {
            auto ai = this->offset[i];
            auto an = this->sizes[i];

            if (!(that[i] >= ai && that[i] - ai < an)) {
                contain = false;
            }
        }

        return contain;
    }

    KMM_HOST_DEVICE
    Range intersection(const Dim<N, T>& that) const {
        return intersection(Range<N, T> {that});
    }

    KMM_HOST_DEVICE
    bool overlaps(const Dim<N, T>& that) const {
        return overlaps(Range<N, T> {that});
    }

    KMM_HOST_DEVICE
    bool contains(const Dim<N, T>& that) const {
        return contains(Range<N, T> {that});
    }

    template<size_t M>
    KMM_HOST_DEVICE Range<N + M> concat(const Range<M>& that) const {
        return {offset.concat(that.offset), sizes.concate(that.sizes)};
    }
};

template<typename... Ts>
KMM_HOST_DEVICE_NOINLINE Point(Ts...) -> Point<sizeof...(Ts)>;

template<typename... Ts>
KMM_HOST_DEVICE_NOINLINE Dim(Ts...) -> Dim<sizeof...(Ts)>;

template<size_t N, typename T>
KMM_HOST_DEVICE_NOINLINE Range(Point<N, T> offset, Dim<N, T> sizes) -> Range<N, T>;

template<size_t N, typename T>
KMM_HOST_DEVICE_NOINLINE Range(Dim<N, T> sizes) -> Range<N, T>;

template<size_t N, typename T>
KMM_HOST_DEVICE bool operator==(const Point<N, T>& a, const Point<N, T>& b) {
    return (const fixed_array<T, N>&)a == (const fixed_array<T, N>&)b;
}

template<size_t N, typename T>
KMM_HOST_DEVICE bool operator!=(const Point<N, T>& a, const Point<N, T>& b) {
    return !(a == b);
}

template<size_t N, typename T>
KMM_HOST_DEVICE bool operator==(const Dim<N, T>& a, const Dim<N, T>& b) {
    return (const fixed_array<T, N>&)a == (const fixed_array<T, N>&)b;
}

template<size_t N, typename T>
KMM_HOST_DEVICE bool operator!=(const Dim<N, T>& a, const Dim<N, T>& b) {
    return !(a == b);
}

template<size_t N, typename T>
KMM_HOST_DEVICE bool operator==(const Range<N, T>& a, const Range<N, T>& b) {
    return a.offset == b.offset && a.sizes == b.sizes;
}

template<size_t N, typename T>
KMM_HOST_DEVICE bool operator!=(const Range<N, T>& a, const Range<N, T>& b) {
    return !(a == b);
}

#define KMM_POINT_OPERATOR_IMPL(OP)                                                       \
    template<size_t N, typename T>                                                        \
    KMM_HOST_DEVICE Point<N, T> operator OP(const Point<N, T>& a, const Point<N, T>& b) { \
        Point<N, T> result;                                                               \
        for (size_t i = 0; is_less(i, N); i++) {                                          \
            result[i] = a[i] OP b[i];                                                     \
        }                                                                                 \
                                                                                          \
        return result;                                                                    \
    }

KMM_POINT_OPERATOR_IMPL(+);
KMM_POINT_OPERATOR_IMPL(-);
KMM_POINT_OPERATOR_IMPL(*);
KMM_POINT_OPERATOR_IMPL(/);

}  // namespace kmm

#include <iostream>

namespace kmm {

template<size_t N, typename T>
std::ostream& operator<<(std::ostream& stream, const Point<N, T>& p) {
    return stream << fixed_array<T, N>(p);
}

template<size_t N, typename T>
std::ostream& operator<<(std::ostream& stream, const Dim<N, T>& p) {
    return stream << fixed_array<T, N>(p);
}

template<size_t N, typename T>
std::ostream& operator<<(std::ostream& stream, const Range<N, T>& p) {
    stream << "{";
    for (size_t i = 0; is_less(i, N); i++) {
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
struct fmt::formatter<kmm::Point<N, T>>: fmt::ostream_formatter {};

template<size_t N, typename T>
struct fmt::formatter<kmm::Dim<N, T>>: fmt::ostream_formatter {};

template<size_t N, typename T>
struct fmt::formatter<kmm::Range<N, T>>: fmt::ostream_formatter {};

#include "kmm/utils/hash_utils.hpp"

template<size_t N, typename T>
struct std::hash<kmm::Point<N, T>>: std::hash<kmm::fixed_array<T, N>> {};

template<size_t N, typename T>
struct std::hash<kmm::Dim<N, T>>: std::hash<kmm::fixed_array<T, N>> {};

template<size_t N, typename T>
struct std::hash<kmm::Range<N, T>> {
    size_t operator()(const kmm::Range<N, T>& p) const {
        kmm::fixed_array<T, N> v[2] = {p.offset, p.sizes};
        return kmm::hash_range(v, v + 2);
    }
};

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "fixed_array.hpp"

#include "kmm/utils/macros.hpp"

namespace kmm {

using default_geometry_type = int64_t;

template<size_t N, typename T = default_geometry_type>
class Index: public fixed_array<T, N> {
  public:
    using storage_type = fixed_array<T, N>;

    KMM_HOST_DEVICE
    explicit constexpr Index(const storage_type& storage) : storage_type(storage) {}

    KMM_HOST_DEVICE
    constexpr Index() {
        for (size_t i = 0; i < N; i++) {
            (*this)[i] = T {};
        }
    }

    template<typename... Ts, typename = typename std::enable_if<(sizeof...(Ts) < N)>::type>
    KMM_HOST_DEVICE Index(T first, Ts&&... args) : Index() {
        (*this)[0] = first;

        size_t index = 0;
        (((*this)[++index] = args), ...);
    }

    template<size_t M, typename U>
    KMM_HOST_DEVICE static constexpr Index from(const fixed_array<U, M>& that) {
        Index result;

        for (size_t i = 0; i < N && is_less(i, M); i++) {
            result[i] = static_cast<T>(that[i]);
        }

        return result;
    }

    KMM_HOST_DEVICE
    static constexpr Index fill(T value) {
        Index result;

        for (size_t i = 0; i < N; i++) {
            result[i] = value;
        }

        return result;
    }

    KMM_HOST_DEVICE
    static constexpr Index zero() {
        return fill(static_cast<T>(0));
    }

    KMM_HOST_DEVICE
    static constexpr Index one() {
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
    KMM_HOST_DEVICE Index<N + M> concat(const Index<M>& that) const {
        fixed_array<T, N + M> result;

        for (size_t i = 0; i < N; i++) {
            result[i] = (*this)[i];
        }

        for (size_t i = 0; is_less(i, M); i++) {
            result[N + i] = that[i];
        }

        return Index<N + M> {result};
    }
};

template<typename T>
class Index<0, T>: public fixed_array<T, 0> {
  public:
    using storage_type = fixed_array<T, 0>;

    KMM_HOST_DEVICE
    explicit constexpr Index(const storage_type& storage) : storage_type(storage) {}

    KMM_HOST_DEVICE
    constexpr Index() {}

    template<size_t M, typename U>
    KMM_HOST_DEVICE static constexpr Index from(const fixed_array<U, M>& that) {
        return {};
    }

    KMM_HOST_DEVICE
    static constexpr Index fill(T value) {
        return {};
    }

    KMM_HOST_DEVICE
    static constexpr Index zero() {
        return {};
    }

    KMM_HOST_DEVICE
    static constexpr Index one() {
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
    KMM_HOST_DEVICE Index<M> concat(const Index<M>& that) const {
        return that;
    }
};

template<size_t N, typename T = default_geometry_type>
class Size: public fixed_array<T, N> {
  public:
    using storage_type = fixed_array<T, N>;

    KMM_HOST_DEVICE
    explicit constexpr Size(const storage_type& storage) : storage_type(storage) {}

    KMM_HOST_DEVICE
    constexpr Size() {
        for (size_t i = 0; i < N; i++) {
            (*this)[i] = static_cast<T>(1);
        }
    }

    template<typename... Ts, typename = typename std::enable_if<(sizeof...(Ts) < N)>::type>
    KMM_HOST_DEVICE Size(T first, Ts&&... args) : Size() {
        (*this)[0] = first;

        size_t index = 0;
        (((*this)[++index] = args), ...);
    }

    KMM_HOST_DEVICE
    static constexpr Size from_point(const Index<N, T>& that) {
        return Size {that};
    }

    template<size_t M, typename U>
    KMM_HOST_DEVICE static constexpr Size from(const fixed_array<U, M>& that) {
        Size result;

        for (size_t i = 0; i < N && is_less(i, M); i++) {
            result[i] = that[i];
        }

        return result;
    }

    KMM_HOST_DEVICE
    static constexpr Size zero() {
        return from_point(Index<N, T>::zero());
    }

    KMM_HOST_DEVICE
    static constexpr Size one() {
        return from_point(Index<N, T>::one());
    }

    KMM_HOST_DEVICE
    Index<N, T> to_point() const {
        return Index<N, T>::from(*this);
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
    Size intersection(const Size& that) const {
        Size<N, T> new_sizes;

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
    bool overlaps(const Size& that) const {
        return !this->is_empty() && !that.is_empty();
    }

    KMM_HOST_DEVICE
    bool contains(const Size& that) const {
        return that.is_empty() || intersection(that) == that;
    }

    KMM_HOST_DEVICE
    bool contains(const Index<N, T>& that) const {
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
    KMM_HOST_DEVICE Size<N + M> concat(const Size<M>& that) const {
        return Size<N + M>(to_point().concat(that.to_point()));
    }
};

template<typename T>
class Size<0, T>: public fixed_array<T, 0> {
  public:
    using storage_type = fixed_array<T, 0>;

    KMM_HOST_DEVICE
    explicit constexpr Size(const storage_type& storage) : storage_type(storage) {}

    KMM_HOST_DEVICE
    constexpr Size() {}

    KMM_HOST_DEVICE
    static constexpr Size from_point(const Index<0, T>& that) {
        return {};
    }

    template<size_t M, typename U>
    KMM_HOST_DEVICE static constexpr Size from(const fixed_array<U, M>& that) {
        return that;
    }

    KMM_HOST_DEVICE
    static constexpr Size zero() {
        return {};
    }

    KMM_HOST_DEVICE
    static constexpr Size one() {
        return {};
    }

    KMM_HOST_DEVICE
    Index<0, T> to_point() const {
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
    Size intersection(const Size& that) const {
        return {};
    }

    KMM_HOST_DEVICE
    bool overlaps(const Size& that) const {
        return true;
    }

    KMM_HOST_DEVICE
    bool contains(const Size& that) const {
        return true;
    }

    KMM_HOST_DEVICE
    bool contains(const Index<0, T>& that) const {
        return true;
    }

    KMM_HOST_DEVICE
    T operator()(size_t axis = 0) const {
        return get(axis);
    }

    template<size_t M>
    KMM_HOST_DEVICE Size<M> concat(const Size<M>& that) const {
        return that;
    }
};

template<size_t N, typename T = default_geometry_type>
class Range {
  public:
    Index<N, T> offset;
    Size<N, T> sizes;

    KMM_HOST_DEVICE
    Range(Index<N, T> offset, Size<N, T> sizes) : offset(offset), sizes(sizes) {}

    KMM_HOST_DEVICE
    Range(Size<N, T> sizes) : Range(Index<N, T>::zero(), sizes) {}

    template<typename... Ts, typename = typename std::enable_if<(sizeof...(Ts) < N)>::type>
    KMM_HOST_DEVICE Range(T first, Ts&&... args) :
        offset(Index<N, T>::zero()),
        sizes(first, args...) {}

    KMM_HOST_DEVICE
    Range() : Range(Size<N, T>::zero()) {}

    KMM_HOST_DEVICE static constexpr Range from_bounds(
        const Index<N, T>& begin,
        const Index<N, T>& end
    ) {
        return {begin, Size<N, T>::from_point(end - begin)};
    }

    template<size_t M, typename U>
    KMM_HOST_DEVICE static constexpr Range from(const Range<M, U>& that) {
        return {Index<N, T>::from(that.offset()), Size<N, T>::from(that.sizes())};
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
    Index<N, T> begin() const {
        return this->offset;
    }

    KMM_HOST_DEVICE
    Index<N, T> end() const {
        Index<N, T> result;
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
        Index<N, T> new_offset;
        Size<N, T> new_sizes;
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
            new_offset = Index<N, T>::zero();
            new_sizes = Size<N, T>::zero();
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
    bool contains(const Index<N, T>& that) const {
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
    Range intersection(const Size<N, T>& that) const {
        return intersection(Range<N, T> {that});
    }

    KMM_HOST_DEVICE
    bool overlaps(const Size<N, T>& that) const {
        return overlaps(Range<N, T> {that});
    }

    KMM_HOST_DEVICE
    bool contains(const Size<N, T>& that) const {
        return contains(Range<N, T> {that});
    }

    template<size_t M>
    KMM_HOST_DEVICE Range<N + M> concat(const Range<M>& that) const {
        return {offset.concat(that.offset), sizes.concate(that.sizes)};
    }
};

template<typename... Ts>
KMM_HOST_DEVICE_NOINLINE Index(Ts...) -> Index<sizeof...(Ts)>;

template<typename... Ts>
KMM_HOST_DEVICE_NOINLINE Size(Ts...) -> Size<sizeof...(Ts)>;

template<size_t N, typename T>
KMM_HOST_DEVICE_NOINLINE Range(Index<N, T> offset, Size<N, T> sizes) -> Range<N, T>;

template<size_t N, typename T>
KMM_HOST_DEVICE_NOINLINE Range(Size<N, T> sizes) -> Range<N, T>;

template<size_t N, typename T>
KMM_HOST_DEVICE bool operator==(const Index<N, T>& a, const Index<N, T>& b) {
    return (const fixed_array<T, N>&)a == (const fixed_array<T, N>&)b;
}

template<size_t N, typename T>
KMM_HOST_DEVICE bool operator!=(const Index<N, T>& a, const Index<N, T>& b) {
    return !(a == b);
}

template<size_t N, typename T>
KMM_HOST_DEVICE bool operator==(const Size<N, T>& a, const Size<N, T>& b) {
    return (const fixed_array<T, N>&)a == (const fixed_array<T, N>&)b;
}

template<size_t N, typename T>
KMM_HOST_DEVICE bool operator!=(const Size<N, T>& a, const Size<N, T>& b) {
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
    KMM_HOST_DEVICE Index<N, T> operator OP(const Index<N, T>& a, const Index<N, T>& b) { \
        Index<N, T> result;                                                               \
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

#include <iosfwd>

namespace kmm {

template<size_t N, typename T>
std::ostream& operator<<(std::ostream& stream, const Index<N, T>& p) {
    return stream << fixed_array<T, N>(p);
}

template<size_t N, typename T>
std::ostream& operator<<(std::ostream& stream, const Size<N, T>& p) {
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
struct fmt::formatter<kmm::Index<N, T>>: fmt::ostream_formatter {};

template<size_t N, typename T>
struct fmt::formatter<kmm::Size<N, T>>: fmt::ostream_formatter {};

template<size_t N, typename T>
struct fmt::formatter<kmm::Range<N, T>>: fmt::ostream_formatter {};

#include "kmm/utils/hash_utils.hpp"

template<size_t N, typename T>
struct std::hash<kmm::Index<N, T>>: std::hash<kmm::fixed_array<T, N>> {};

template<size_t N, typename T>
struct std::hash<kmm::Size<N, T>>: std::hash<kmm::fixed_array<T, N>> {};

template<size_t N, typename T>
struct std::hash<kmm::Range<N, T>> {
    size_t operator()(const kmm::Range<N, T>& p) const {
        kmm::fixed_array<T, N> v[2] = {p.offset, p.sizes};
        return kmm::hash_range(v, v + 2);
    }
};

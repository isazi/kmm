#pragma once

#include <limits>

#include "partition.hpp"

#include "kmm/core/geometry.hpp"
#include "kmm/utils/checked_math.hpp"
#include "kmm/utils/integer_fun.hpp"

namespace kmm {

struct FullMapping {
    template<size_t N, size_t M>
    rect<M> operator()(Chunk<N> chunk, dim<M> bounds) {
        return bounds;
    }
};

static constexpr FullMapping full = {};

struct OneToOneMapping {
    template<size_t N, size_t M>
    rect<M> operator()(Chunk<N> chunk, dim<M> bounds) {
        point<M> offset = point<M>::zero();
        dim<M> shape = bounds;

        for (size_t i = 0; i < N && i < M; i++) {
            offset[i] = chunk.offset(i);
            shape[i] = chunk.size(i);
        }

        return {offset, shape};
    }
};

static constexpr OneToOneMapping one_to_one = {};

struct VariableMapping {
    explicit constexpr VariableMapping(size_t axis) : axis(axis) {}

    template<size_t N, size_t M>
    rect<M> operator()(Chunk<N> chunk, dim<M> bounds) {
        point<M> offset = point<M>::zero();
        dim<M> sizes = bounds;

        if constexpr (M > 0) {
            offset[0] = chunk.local_size.offset.get(axis);
            sizes[0] = chunk.local_size.sizes.get(axis);
        }

        return {offset, sizes};
    }

    size_t axis;
};

/**
 * Represents the range from `(scale * variable + offset)/divisor` to
 * `(scale * variable + offset + length)/divisor` (not inclusive).
 */
struct IndexMapping {
    constexpr IndexMapping(
        VariableMapping variable,
        int64_t scale = 1,
        int64_t offset = 0,
        int64_t length = 1,
        int64_t divisor = 1) :
        variable(variable),
        scale(scale),
        offset(offset),
        length(length),
        divisor(divisor) {}

    IndexMapping negate() const;
    IndexMapping scale_by(int64_t f) const;
    IndexMapping offset_by(int64_t f) const;
    IndexMapping divide_by(int64_t f) const;

    template<size_t N>
    rect<1> apply(Chunk<N> chunk) const {
        auto m = chunk.local_size;

        auto a = div_floor(m.begin(variable.axis) * scale + offset, divisor);
        auto b = div_floor((m.end(variable.axis) - 1) * scale + (offset + length), divisor) + 1;
        return {{a}, {b - a}};
    }

    VariableMapping variable;
    int64_t scale = 1;
    int64_t offset = 0;
    int64_t length = 1;
    int64_t divisor = 1;
};

namespace placeholders {
static constexpr VariableMapping _x = VariableMapping(0);
static constexpr VariableMapping _y = VariableMapping(1);
static constexpr VariableMapping _z = VariableMapping(2);
static constexpr VariableMapping _w = VariableMapping(3);

static constexpr VariableMapping _i = VariableMapping(0);
static constexpr VariableMapping _j = VariableMapping(1);
static constexpr VariableMapping _k = VariableMapping(2);

static constexpr VariableMapping _0 = VariableMapping(0);
static constexpr VariableMapping _1 = VariableMapping(1);
static constexpr VariableMapping _2 = VariableMapping(2);
static constexpr VariableMapping _3 = VariableMapping(3);

static constexpr FullMapping _ = full;
}  // namespace placeholders

inline IndexMapping operator+(IndexMapping m, int64_t f) {
    return m.offset_by(f);
}

inline IndexMapping operator+(int64_t f, IndexMapping m) {
    return m.offset_by(f);
}

inline IndexMapping operator-(IndexMapping m) {
    return m.negate();
}

inline IndexMapping operator-(IndexMapping m, int64_t f) {
    return m + (-f);
}

inline IndexMapping operator-(int64_t f, IndexMapping m) {
    return f + (-m);
}

inline IndexMapping operator*(IndexMapping m, int64_t f) {
    return m.scale_by(f);
}

inline IndexMapping operator*(int64_t f, IndexMapping m) {
    return m.scale_by(f);
}

inline IndexMapping operator/(IndexMapping m, int64_t f) {
    return m.divide_by(f);
}

namespace detail {
template<typename T>
struct IntoIndexMapping {};

template<>
struct IntoIndexMapping<IndexMapping> {
    static IndexMapping call(IndexMapping v) {
        return v;
    }
};

template<>
struct IntoIndexMapping<VariableMapping> {
    static IndexMapping call(VariableMapping v) {
        return v;
    }
};

template<>
struct IntoIndexMapping<FullMapping> {
    static IndexMapping call(FullMapping v) {
        return {VariableMapping(0), 0, 0, std::numeric_limits<int64_t>::max()};
    }
};

#define KMM_IMPL_INTO_INDEX_MAPPING_FOR_INT(T)                        \
    template<>                                                        \
    struct IntoIndexMapping<T> {                                      \
        static IndexMapping call(T v) {                               \
            return {VariableMapping(0), 0, checked_cast<int64_t>(v)}; \
        }                                                             \
    };

KMM_IMPL_INTO_INDEX_MAPPING_FOR_INT(unsigned long long)
KMM_IMPL_INTO_INDEX_MAPPING_FOR_INT(unsigned long)
KMM_IMPL_INTO_INDEX_MAPPING_FOR_INT(unsigned int)
KMM_IMPL_INTO_INDEX_MAPPING_FOR_INT(unsigned short)

KMM_IMPL_INTO_INDEX_MAPPING_FOR_INT(signed long long)
KMM_IMPL_INTO_INDEX_MAPPING_FOR_INT(signed long)
KMM_IMPL_INTO_INDEX_MAPPING_FOR_INT(signed int)
KMM_IMPL_INTO_INDEX_MAPPING_FOR_INT(signed short)

KMM_IMPL_INTO_INDEX_MAPPING_FOR_INT(unsigned char)
KMM_IMPL_INTO_INDEX_MAPPING_FOR_INT(signed char)
KMM_IMPL_INTO_INDEX_MAPPING_FOR_INT(char)
KMM_IMPL_INTO_INDEX_MAPPING_FOR_INT(bool)
}  // namespace detail

template<typename T>
IndexMapping into_index_mapping(T&& that) {
    return detail::IntoIndexMapping<std::decay_t<T>>::call(that);
}

template<size_t N>
struct SliceMapping {
    template<size_t M>
    rect<N> operator()(Chunk<M> chunk, dim<N> bounds) {
        point<N> offset;
        dim<N> sizes;

        for (size_t i = 0; i < N; i++) {
            auto ab = mappings[i].apply(chunk);
            offset[i] = ab.offset;
            sizes[i] = ab.sizes;
        }

        return rect<N> {offset, sizes}.intersection(bounds);
    }

    IndexMapping mappings[N];
};

template<>
struct SliceMapping<0> {
    template<size_t M>
    rect<0> operator()(Chunk<M> chunk, dim<0> bounds) {
        return {};
    }
};

template<typename... Ts>
SliceMapping<sizeof...(Ts)> slice(Ts... indices) {
    return {into_index_mapping(indices)...};
}

}  // namespace kmm
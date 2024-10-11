#pragma once

#include "partition.hpp"
#include "spdlog/spdlog.h"

#include "kmm/core/geometry.hpp"
#include "kmm/core/reduction.hpp"
#include "kmm/utils/integer_fun.hpp"

namespace kmm {

struct All {
    template<size_t N>
    Rect<N> operator()(TaskChunk chunk, Rect<N> bounds) const {
        return bounds;
    }
};

struct Axis {
    constexpr Axis() : m_axis(0) {}
    explicit constexpr Axis(size_t axis) : m_axis(axis) {}

    Rect<1> operator()(TaskChunk chunk) const {
        return Rect<1> {chunk.offset.get(m_axis), chunk.size.get(m_axis)};
    }

    Rect<1> operator()(TaskChunk chunk, Rect<1> bounds) const {
        return (*this)(chunk).intersection(bounds);
    }

    size_t get() const {
        return m_axis;
    }

    operator size_t() const {
        return get();
    }

  private:
    size_t m_axis = 0;
};

struct IdentityMap {
    template<size_t N>
    Rect<N> operator()(TaskChunk chunk, Rect<N> bounds) const {
        return {Point<N>::from(chunk.offset), Dim<N>::from(chunk.size)};
    }
};

namespace placeholders {
static constexpr All _;

static constexpr Axis _x = Axis(0);
static constexpr Axis _y = Axis(1);
static constexpr Axis _z = Axis(2);

static constexpr Axis _i = Axis(0);
static constexpr Axis _j = Axis(1);
static constexpr Axis _k = Axis(2);

static constexpr Axis _0 = Axis(0);
static constexpr Axis _1 = Axis(1);
static constexpr Axis _2 = Axis(2);

static constexpr IdentityMap one_to_one;
static constexpr All all;
}  // namespace placeholders

// (scale * variable + offset + [0...length]) / divisor
struct IndexMap {
    IndexMap(
        Axis variable = {},
        int64_t scale = 1,
        int64_t offset = 0,
        int64_t length = 1,
        int64_t divisor = 1
    );

    static IndexMap range(IndexMap begin, IndexMap end);
    IndexMap offset_by(int64_t offset) const;
    IndexMap scale_by(int64_t factor) const;
    IndexMap divide_by(int64_t divisor) const;
    IndexMap negate() const;
    Rect<1> apply(TaskChunk chunk) const;

    Rect<1> operator()(TaskChunk chunk) const {
        return apply(chunk);
    }

    Rect<1> operator()(TaskChunk chunk, Rect<1> bounds) const {
        return apply(chunk).intersection(bounds);
    }

    friend std::ostream& operator<<(std::ostream& f, const IndexMap& that);

  private:
    Axis m_variable;
    int64_t m_scale;
    int64_t m_offset;
    int64_t m_length;
    int64_t m_divisor;
};

inline IndexMap range(IndexMap begin, IndexMap end) {
    return IndexMap::range(begin, end);
}

inline IndexMap range(int64_t begin, int64_t end) {
    return {Axis {}, 0, begin, end - begin};
}

inline IndexMap range(int64_t end) {
    return range(0, end);
}

inline IndexMap operator+(IndexMap a) {
    return a;
}

inline IndexMap operator+(IndexMap a, int64_t b) {
    return a.offset_by(b);
}

inline IndexMap operator+(int64_t a, IndexMap b) {
    return b.offset_by(a);
}

inline IndexMap operator-(IndexMap a) {
    return a.negate();
}

inline IndexMap operator-(IndexMap a, int64_t b) {
    return a + (-b);
}

inline IndexMap operator-(int64_t a, IndexMap b) {
    return a + (-b);
}

inline IndexMap operator*(IndexMap a, int64_t b) {
    return a.scale_by(b);
}

inline IndexMap operator*(int64_t a, IndexMap b) {
    return b.scale_by(a);
}

inline IndexMap operator/(IndexMap a, int64_t b) {
    return a.divide_by(b);
}

template<size_t N>
struct MultiIndexMap {
    Rect<N> operator()(TaskChunk chunk) const {
        Rect<N> result;

        for (size_t i = 0; i < N; i++) {
            Rect<1> range = (this->axes[i])(chunk);
            result.offset[i] = range.offset[0];
            result.sizes[i] = range.sizes[0];
        }

        return result;
    }

    Rect<N> operator()(TaskChunk chunk, Rect<N> bounds) const {
        Rect<N> result;

        for (size_t i = 0; i < N; i++) {
            Rect<1> range = (this->axes[i])(chunk, Rect<1> {bounds.offset[i], bounds.sizes[i]});

            result.offset[i] = range.offset[0];
            result.sizes[i] = range.sizes[0];
        }

        return result;
    }

    IndexMap axes[N];
};

template<>
struct MultiIndexMap<0> {
    Rect<0> operator()(TaskChunk chunk, Rect<0> bounds = {}) const {
        return {};
    }
};

inline IndexMap into_index_map(int64_t m) {
    return {Axis {}, 0, m};
}

inline IndexMap into_index_map(Axis m) {
    return m;
}

inline IndexMap into_index_map(IndexMap m) {
    return m;
}

inline IndexMap into_index_map(All m) {
    return {Axis(), 0, 0, std::numeric_limits<int64_t>::max()};
}

template<typename... Is>
MultiIndexMap<sizeof...(Is)> slice(const Is&... slices) {
    return {into_index_map(slices)...};
}

template<typename... Is>
MultiIndexMap<sizeof...(Is)> tile(const Is&... length) {
    size_t variable = 0;
    return {IndexMap(
        Axis {variable++},
        checked_cast<int64_t>(length),
        0,
        checked_cast<int64_t>(length)
    )...};
}

namespace detail {
template<typename T>
struct MapperDimensionality: std::integral_constant<size_t, 0> {};

template<size_t N>
struct MapperDimensionality<Rect<N>>: std::integral_constant<size_t, N> {};
}  // namespace detail

template<typename F>
static constexpr size_t mapper_dimensionality =
    detail::MapperDimensionality<std::invoke_result_t<F, TaskChunk>>::value;

template<typename F, size_t N>
static constexpr bool is_dimensionality_accepted_by_mapper =
    detail::MapperDimensionality<std::invoke_result_t<F, TaskChunk, Rect<N>>>::value == N;

}  // namespace kmm

template<>
struct fmt::formatter<kmm::IndexMap>: fmt::ostream_formatter {};
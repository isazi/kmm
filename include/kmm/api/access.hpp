#pragma once

#include "partition.hpp"
#include "spdlog/spdlog.h"

#include "kmm/core/geometry.hpp"
#include "kmm/utils/integer_fun.hpp"

namespace kmm {

struct IdentityMapping {
    template<size_t N>
    rect<N> operator()(Chunk chunk, rect<N> bounds) const {
        return {chunk.offset, chunk.size};
    }
};

static constexpr IdentityMapping one_to_one;

struct FullMapping {
    template<size_t M>
    rect<M> operator()(Chunk chunk, rect<M> bounds) const {
        return bounds;
    }
};

struct AxesMapping {
    constexpr AxesMapping() : m_axis(0) {}
    explicit constexpr AxesMapping(size_t axis) : m_axis(axis) {}

    template<size_t M>
    rect<M> operator()(Chunk chunk, rect<M> bounds) const {
        rect<M> result = bounds;

        if (M > 0) {
            result.offset[0] = chunk.offset.get(m_axis);
            result.sizes[0] = chunk.size.get(m_axis);
        }

        return result;
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

namespace placeholders {
static constexpr FullMapping _;

static constexpr AxesMapping _x = AxesMapping(0);
static constexpr AxesMapping _y = AxesMapping(1);
static constexpr AxesMapping _z = AxesMapping(2);

static constexpr AxesMapping _i = AxesMapping(0);
static constexpr AxesMapping _j = AxesMapping(1);
static constexpr AxesMapping _k = AxesMapping(2);

static constexpr AxesMapping _0 = AxesMapping(0);
static constexpr AxesMapping _1 = AxesMapping(1);
static constexpr AxesMapping _2 = AxesMapping(2);

}  // namespace placeholders

// (scale * variable + offset + [0...length]) / divisor
struct IndexMapping {
    IndexMapping(
        AxesMapping variable = {},
        int64_t scale = 1,
        int64_t offset = 0,
        int64_t length = 1,
        int64_t divisor = 1);

    static IndexMapping range(IndexMapping begin, IndexMapping end);
    IndexMapping offset_by(int64_t offset) const;
    IndexMapping scale_by(int64_t factor) const;
    IndexMapping divide_by(int64_t divisor) const;
    IndexMapping negate() const;

    template<size_t M>
    rect<M> operator()(Chunk chunk, rect<M> bounds) const {
        rect<M> result = bounds;

        if (M > 0) {
            int64_t a0 = chunk.offset.get(m_variable);
            int64_t a1 = a0 + chunk.size.get(m_variable) - 1;

            int64_t b0;
            int64_t b1;

            if (m_scale > 0) {
                b0 = m_scale * a0 + m_offset;
                b1 = m_scale * a1 + m_offset + m_length - 1;
            } else if (m_scale < 0) {
                b0 = m_scale * a1 + m_offset;
                b1 = m_scale * a0 + m_offset + m_length - 1;
            } else {
                b0 = m_offset;
                b1 = m_offset + m_length - 1;
            }

            if (m_divisor != 1) {
                b0 = div_floor(b0, m_divisor);
                b1 = div_floor(b1, m_divisor);
            }

            int64_t i = b0;
            int64_t n = b1 - b0 + 1;

            result.offset[0] = i;
            result.sizes[0] = n;
        }

        return result.intersection(bounds);
    }

    friend std::ostream& operator<<(std::ostream& f, const IndexMapping& that);

  private:
    AxesMapping m_variable;
    int64_t m_scale;
    int64_t m_offset;
    int64_t m_length;
    int64_t m_divisor;
};

inline IndexMapping range(IndexMapping begin, IndexMapping end) {
    return IndexMapping::range(begin, end);
}

inline IndexMapping range(int64_t begin, int64_t end) {
    return {AxesMapping {}, 0, begin, end - begin};
}

inline IndexMapping range(int64_t end) {
    return range(0, end);
}

inline IndexMapping operator+(IndexMapping a) {
    return a;
}

inline IndexMapping operator+(IndexMapping a, int64_t b) {
    return a.offset_by(b);
}

inline IndexMapping operator+(int64_t a, IndexMapping b) {
    return b.offset_by(a);
}

inline IndexMapping operator-(IndexMapping a) {
    return a.negate();
}

inline IndexMapping operator-(IndexMapping a, int64_t b) {
    return a + (-b);
}

inline IndexMapping operator-(int64_t a, IndexMapping b) {
    return a + (-b);
}

inline IndexMapping operator*(IndexMapping a, int64_t b) {
    return a.scale_by(b);
}

inline IndexMapping operator*(int64_t a, IndexMapping b) {
    return b.scale_by(a);
}

inline IndexMapping operator/(IndexMapping a, int64_t b) {
    return a.divide_by(b);
}

inline IndexMapping into_index_mapping(int64_t m) {
    return {AxesMapping {}, 0, m};
}

inline IndexMapping into_index_mapping(AxesMapping m) {
    return m;
}

inline IndexMapping into_index_mapping(IndexMapping m) {
    return m;
}

inline IndexMapping into_index_mapping(FullMapping m) {
    return {AxesMapping(), 0, 0, std::numeric_limits<int64_t>::max()};
}

template<size_t N>
struct SliceMapping {
    rect<N> operator()(Chunk chunk, rect<N> bounds) const {
        rect<N> result;

        for (size_t i = 0; i < N; i++) {
            rect<1> range = (indices[i])(chunk, rect<1> {bounds.offset[i], bounds.sizes[i]});

            result.offset[i] = range.offset[0];
            result.sizes[i] = range.sizes[0];
        }

        return result;
    }

    IndexMapping indices[N];
};

template<>
struct SliceMapping<0> {
    rect<0> operator()(Chunk chunk, rect<0> bounds) const {
        return {};
    }
};

template<typename... Is>
SliceMapping<sizeof...(Is)> slice(const Is&... slices) {
    return {into_index_mapping(slices)...};
}

template<typename... Is>
SliceMapping<sizeof...(Is)> tile(const Is&... length) {
    size_t variable = 0;
    return {IndexMapping(
        AxesMapping {variable++},
        checked_cast<int64_t>(length),
        0,
        checked_cast<int64_t>(length))...};
}

template<typename T, typename I = FullMapping>
struct Read {
    T argument;
    I index_mapping;
};

template<typename I = FullMapping, typename T>
Read<T, I> read(T argument, I index_mapping = {}) {
    return {argument, index_mapping};
}

template<typename T, typename I = FullMapping>
struct Write {
    T& argument;
    I index_mapping;
};

template<typename I = FullMapping, typename T>
Write<T, I> write(T& argument, I index_mapping = {}) {
    return {argument, index_mapping};
}

template<typename T, typename I = FullMapping>
struct Reduce {
    T& argument;
    ReductionOp op;
    I index_mapping;
};

template<typename I = FullMapping, typename T>
Reduce<T, I> reduce(T& argument, ReductionOp op, I index_mapping = {}) {
    return {argument, op, index_mapping};
}

}  // namespace kmm

template<>
struct fmt::formatter<kmm::IndexMapping>: fmt::ostream_formatter {};
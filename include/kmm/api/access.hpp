#pragma once

#include "partition.hpp"
#include "spdlog/spdlog.h"

#include "kmm/core/geometry.hpp"

namespace kmm {

struct IdentityMapping {
    template<size_t N>
    rect<N> operator()(Chunk<N> chunk, rect<N> bounds) const {
        return {chunk.offset, chunk.size};
    }
};

static constexpr IdentityMapping one_to_one;

struct FullMapping {
    template<size_t N, size_t M>
    rect<M> operator()(Chunk<N> chunk, rect<M> bounds) const {
        return bounds;
    }
};

struct AxesMapping {
    constexpr AxesMapping() : m_axis(0) {}
    explicit constexpr AxesMapping(size_t axis) : m_axis(axis) {}

    template<size_t N, size_t M>
    rect<M> operator()(Chunk<N> chunk, rect<M> bounds) const {
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
static constexpr AxesMapping _3 = AxesMapping(3);
static constexpr AxesMapping _4 = AxesMapping(4);

}  // namespace placeholders

struct IndexMapping {
    constexpr IndexMapping(
        AxesMapping variable = {},
        int64_t scale = 1,
        int64_t offset = 0,
        int64_t length = 1) :
        m_variable(variable),
        m_scale(scale),
        m_offset(offset),
        m_length(length) {}

    IndexMapping offset_by(int64_t offset) const;
    IndexMapping scale_by(int64_t factor) const;
    IndexMapping negate() const;

    template<size_t N, size_t M>
    rect<M> operator()(Chunk<N> chunk, rect<M> bounds) const {
        rect<M> result = bounds;

        if (M > 0) {
            int64_t i;
            int64_t n;

            if (m_scale > 0) {
                i = chunk.offset.get(m_variable) * m_scale + m_offset;
                n = (chunk.size.get(m_variable) - 1) * m_scale + 1;
            } else if (m_scale < 0) {
                i = (chunk.offset.get(m_variable) + chunk.size.get(m_variable) - 1) * m_scale
                    + m_offset;
                n = (chunk.size.get(m_variable) - 1) * -m_scale + 1;
            } else {
                i = m_offset;
                n = 1;
            }

            spdlog::debug(
                "scale={} offset={} length={} maps={},{} to {},{}",
                m_scale,
                m_offset,
                m_length,
                chunk.offset.get(m_variable),
                chunk.size.get(m_variable),
                i,
                n);

            result.offset[0] = i;
            result.sizes[0] = n;
        }

        spdlog::debug("{} intersection {} result {}", result, bounds, result.intersection(bounds));
        return result.intersection(bounds);
    }

  private:
    AxesMapping m_variable;
    int64_t m_scale;
    int64_t m_offset;
    int64_t m_length;
};

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
    template<size_t M>
    rect<N> operator()(Chunk<M> chunk, rect<N> bounds) const {
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
    template<size_t M>
    rect<0> operator()(Chunk<M> chunk, rect<0> bounds) const {
        return {};
    }
};

template<typename... Is>
SliceMapping<sizeof...(Is)> slice(const Is&... slices) {
    return {into_index_mapping(slices)...};
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

}  // namespace kmm
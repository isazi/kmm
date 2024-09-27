#include <cstdint>

#include "kmm/api/access.hpp"
#include "kmm/utils/checked_math.hpp"

namespace kmm {

int64_t gcd(int64_t a, int64_t b) {
    if (a == 0) {
        return b;
    } else {
        return gcd(b % a, a);
    }
}

IndexMapping::IndexMapping(
    AxesMapping variable,
    int64_t scale,
    int64_t offset,
    int64_t length,
    int64_t divisor) :
    m_variable(variable),
    m_scale(scale),
    m_offset(offset),
    m_length(length),
    m_divisor(divisor) {
    if (m_length < 0) {
        m_length = 0;
    }

    if (m_divisor < 0) {
        m_scale = -m_scale;
        m_offset = -checked_add(m_offset, m_length - 1);
        m_divisor = -m_divisor;
    }

    if (m_scale != 1 && m_divisor != 1) {
        auto common = gcd(gcd(abs(m_scale), abs(m_offset)), gcd(m_length, m_divisor));

        if (common != 1) {
            m_scale /= common;
            m_offset /= common;
            m_length /= common;
            m_divisor /= common;
        }
    }
}

IndexMapping IndexMapping::range(IndexMapping begin, IndexMapping end) {
    if (begin.m_scale != end.m_scale || begin.m_divisor != end.m_divisor) {
        throw std::runtime_error(fmt::format(
            "`range` can only be created over two expressions with the same scaling factor: `{}` and `{}`",
            begin,
            end));
    }

    if (begin.m_variable != end.m_variable && begin.m_scale != 0) {
        throw std::runtime_error(fmt::format(
            "`range` can only be created over two expressions with the same variable: `{}` and `{}`",
            begin,
            end));
    }

    return {
        begin.m_variable,
        begin.m_scale,
        begin.m_offset,
        (end.m_offset - begin.m_offset) + end.m_length,
        begin.m_divisor};
}

IndexMapping IndexMapping::offset_by(int64_t offset) const {
    auto new_offset = checked_add(m_offset, checked_mul(m_divisor, offset));
    return {m_variable, m_scale, new_offset, m_length, m_divisor};
}

IndexMapping IndexMapping::scale_by(int64_t factor) const {
    if (factor < 0) {
        return negate().scale_by(-factor);
    }

    return {
        m_variable,
        checked_mul(m_scale, factor),
        checked_mul(m_offset, factor),
        checked_mul(m_length - 1, factor) + 1,
        m_divisor};
}

IndexMapping IndexMapping::divide_by(int64_t divisor) const {
    if (divisor < 0) {
        return negate().divide_by(-divisor);
    }

    return {m_variable, m_scale, m_offset, m_length, checked_mul(m_divisor, divisor)};
}

IndexMapping IndexMapping::negate() const {
    return {m_variable, -m_scale, -checked_add(m_offset, m_length - 1), m_length, m_divisor};
}

rect<1> IndexMapping::apply(Chunk chunk) const {
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

    rect<1> result;
    result.offset[0] = i;
    result.sizes[0] = n;

    return result;
}

static void write_mapping(
    std::ostream& f,
    AxesMapping v,
    int64_t scale,
    int64_t offset,
    int64_t divisor) {
    static constexpr const char* variables[] = {"x", "y", "z", "w", "v4", "v5", "v6", "v7"};
    const char* var = v < 8 ? variables[v] : "?";

    if (scale != 1) {
        if (offset != 0) {
            f << "(" << scale << "*" << var << " + " << offset << ")";
        } else {
            f << scale << "*" << var;
        }
    } else {
        if (offset != 0) {
            f << "(" << var << " + " << offset << ")";
        } else {
            f << var;
        }
    }

    if (divisor != 1) {
        f << "/" << divisor;
    }
}

std::ostream& operator<<(std::ostream& f, const IndexMapping& that) {
    write_mapping(f, that.m_variable, that.m_scale, that.m_offset, that.m_divisor);

    if (that.m_length != 1) {
        f << "...";
        write_mapping(
            f,
            that.m_variable,
            that.m_scale,
            that.m_offset + that.m_length,
            that.m_divisor);
    }

    return f;
}

}  // namespace kmm
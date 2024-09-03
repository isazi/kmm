#include <cstdint>

#include "kmm/api/access.hpp"
#include "kmm/utils/checked_math.hpp"

namespace kmm {

IndexMapping IndexMapping::offset_by(int64_t offset) const {
    return {m_variable, m_scale, checked_add(m_offset, offset), m_length};
}

IndexMapping IndexMapping::scale_by(int64_t factor) const {
    if (factor < 0) {
        return negate().scale_by(-factor);
    }

    return {
        m_variable,
        checked_mul(m_scale, factor),
        checked_mul(m_offset, factor),
        checked_mul(m_length - 1, factor) + 1};
}

IndexMapping IndexMapping::negate() const {
    return {m_variable, -m_scale, -checked_add(m_offset, m_length - 1), m_length};
}
}  // namespace kmm
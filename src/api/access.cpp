#include <stdexcept>

#include "kmm/api/access.hpp"
#include "kmm/utils/integer_fun.hpp"

namespace kmm {

IndexMapping IndexMapping::negate() const {
    return IndexMapping {variable, -scale, -offset, length, divisor}.offset_by(1 - length);
}

IndexMapping IndexMapping::scale_by(int64_t f) const {
    if (f < 0) {
        return this->negate().scale_by(-f);
    }

    if (f == 0) {
        return {variable, 0, 0, 0, 1};
    }

    return {variable, scale * f, offset * f, length * f - (f - 1), divisor};
}

IndexMapping IndexMapping::offset_by(int64_t f) const {
    return {variable, scale, offset + f * divisor, length, divisor};
}

IndexMapping IndexMapping::divide_by(int64_t f) const {
    if (f < 0) {
        return this->negate().divide_by(-f);
    }

    return {variable, scale, offset, length, divisor * f};
}

}  // namespace kmm
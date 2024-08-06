#include <stdexcept>

#include "kmm/utils/checked_math.hpp"

namespace kmm {

void throw_overflow_exception() {
    throw std::overflow_error("operation resulted in overflow");
}

}  // namespace kmm
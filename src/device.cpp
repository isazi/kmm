#include "kmm/device.hpp"
#include "kmm/runtime.hpp"
#include "kmm/runtime_impl.hpp"

namespace kmm {

InvalidDeviceException::InvalidDeviceException(
    const std::type_info& expected,
    const std::type_info& gotten) {
    m_message = fmt::format(
        "invalid device: expecting an device of type `{}`, but gotten an device of type `{}`",
        expected.name(),
        gotten.name());
}

const char* InvalidDeviceException::what() const noexcept {
    return m_message.c_str();
}
}  // namespace kmm

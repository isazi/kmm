#include "fmt/format.h"

#include "kmm/core/task.hpp"

namespace kmm {

InvalidExecutionContext::InvalidExecutionContext(
    const std::type_info& expected,
    const std::type_info& gotten
) {
    m_message = fmt::format(
        "task expected an execution context of type {}, but was executed with type {}",
        expected.name(),
        gotten.name()
    );
}

const char* InvalidExecutionContext::what() const noexcept {
    return m_message.c_str();
}

}  // namespace kmm
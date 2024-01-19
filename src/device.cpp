#include "kmm/device.hpp"
#include "kmm/runtime_impl.hpp"

namespace kmm {
size_t TaskRequirements::add_input(BlockId block_id) {
    size_t n = inputs.size();
    for (size_t i = 0; i < n; i++) {
        if (inputs[i].block_id == block_id) {
            return i;
        }
    }

    inputs.push_back(TaskInput {.block_id = block_id, .memory_id = std::nullopt});

    return n;
}

size_t TaskRequirements::add_input(BlockId block_id, MemoryId memory_id) {
    size_t n = inputs.size();
    for (size_t i = 0; i < n; i++) {
        if (inputs[i].block_id == block_id) {
            if (inputs[i].memory_id == memory_id || inputs[i].memory_id == std::nullopt) {
                inputs[i].memory_id = memory_id;
                return i;
            }
        }
    }

    inputs.push_back(TaskInput {.block_id = block_id, .memory_id = memory_id});

    return n;
}

size_t TaskRequirements::add_input(BlockId block_id, RuntimeImpl& rt) {
    auto memory_id = rt.device_info(device_id).memory_affinity();
    return add_input(block_id, memory_id);
}

size_t TaskRequirements::add_output(std::unique_ptr<BlockHeader> header, MemoryId memory_id) {
    size_t n = outputs.size();
    outputs.push_back(TaskOutput {.header = std::move(header), .memory_id = memory_id});

    return n;
}

size_t TaskRequirements::add_output(std::unique_ptr<BlockHeader> header, RuntimeImpl& rt) {
    auto memory_id = rt.device_info(device_id).memory_affinity();
    return add_output(std::move(header), memory_id);
}

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

#include "kmm/runtime_handle.hpp"
#include "kmm/task.hpp"

namespace kmm {

TaskBuilder::TaskBuilder(Runtime* runtime, DeviceId id) : m_runtime(runtime), m_requirements(id) {}

EventId TaskBuilder::submit(std::shared_ptr<Task> task) {
    auto event_id = m_runtime->submit_task(std::move(task), std::move(m_requirements));
    auto rt = RuntimeHandle(m_runtime->shared_from_this());

    for (auto& callback : m_callbacks) {
        callback->call(event_id);
    }

    return event_id;
}

std::shared_ptr<Runtime> TaskBuilder::runtime() const {
    return m_runtime->shared_from_this();
}

size_t TaskBuilder::add_input(BlockId block_id) {
    auto memory_id = m_runtime->device_info(m_requirements.device_id).memory_affinity();
    return add_input(block_id, memory_id);
}

size_t TaskBuilder::add_input(BlockId block_id, MemoryId memory_id) {
    size_t n = m_requirements.inputs.size();
    for (size_t i = 0; i < n; i++) {
        auto& input = m_requirements.inputs[i];

        if (input.block_id == block_id) {
            if (input.memory_id == memory_id || input.memory_id == std::nullopt) {
                input.memory_id = memory_id;
                return i;
            }
        }
    }

    m_requirements.inputs.push_back(TaskInput {.block_id = block_id, .memory_id = memory_id});
    return n;
}

size_t TaskBuilder::add_output(std::unique_ptr<BlockHeader> header, MemoryId memory_id) {
    size_t n = m_requirements.outputs.size();
    m_requirements.outputs.push_back(
        TaskOutput {.header = std::move(header), .memory_id = memory_id});

    return n;
}

size_t TaskBuilder::add_output(std::unique_ptr<BlockHeader> header) {
    auto memory_id = m_runtime->device_info(m_requirements.device_id).memory_affinity();
    return add_output(std::move(header), memory_id);
}

}  // namespace kmm
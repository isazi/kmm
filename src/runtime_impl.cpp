#include "kmm/runtime_impl.hpp"

#include "kmm/worker.hpp"

namespace kmm {

VirtualBufferId RuntimeImpl::create_buffer(const BufferDescription& spec) const {
    std::lock_guard guard {m_mutex};
    VirtualBufferId id = m_buffer_manager.create_buffer(spec);

    auto cmd = CommandBufferCreate {
        .id = BufferId(id),
        .description = spec,
    };

    m_worker->submit_command(m_next_event_id++, cmd);
    return id;
}

void RuntimeImpl::increment_buffer_references(VirtualBufferId id, uint64_t count) const {
    std::lock_guard guard {m_mutex};
    m_buffer_manager.increment_buffer_references(id, count);
}

void RuntimeImpl::decrement_buffer_references(VirtualBufferId id, uint64_t count) const {
    std::lock_guard guard {m_mutex};
    bool deleted = m_buffer_manager.decrement_buffer_references(id, count);

    if (deleted) {
        auto cmd = CommandBufferDelete {BufferId(id)};
        m_worker->submit_command(m_next_event_id++, cmd);
    }
}

JobId RuntimeImpl::submit_task(
    DeviceId device_id,
    std::shared_ptr<Task> task,
    std::vector<VirtualBufferRequirement> virtual_buffers,
    std::vector<JobId> dependencies) const {
    std::lock_guard guard {m_mutex};

    JobId task_id = m_next_event_id++;
    std::vector<BufferRequirement> buffers;

    for (const auto& req : virtual_buffers) {
        buffers.push_back(m_buffer_manager.update_buffer_access(
            req.buffer_id,  //
            task_id,
            req.mode,
            dependencies));
    }

    CommandExecute cmd {
        .device_id = device_id,
        .task = std::move(task),
    };

    m_worker->submit_command(task_id, std::move(cmd), dependencies, buffers);
    return task_id;
}
}  // namespace kmm
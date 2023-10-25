#include "kmm/runtime_impl.hpp"

#include "kmm/worker.hpp"

namespace kmm {

VirtualBufferId RuntimeImpl::create_buffer(const BufferDescription& spec) const {
    std::lock_guard guard {m_mutex};
    VirtualBufferId id = m_buffer_manager.create_buffer(spec);
    m_worker->create_buffer(BufferId(id), spec);
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
        m_worker->delete_buffer(BufferId(id), {});
    }
}

TaskId RuntimeImpl::submit_task(
    DeviceId device_id,
    std::shared_ptr<Task> task,
    std::vector<VirtualBufferRequirement> virtual_buffers,
    std::vector<TaskId> dependencies) const {
    std::lock_guard guard {m_mutex};

    TaskId task_id = m_next_task_id;
    m_next_task_id = task_id + 1;

    std::vector<BufferRequirement> buffers;

    for (const auto& req : virtual_buffers) {
        buffers.push_back(m_buffer_manager.update_buffer_access(
            req.buffer_id,  //
            task_id,
            req.mode,
            dependencies));
    }

    m_worker->insert_task(task_id, device_id, task, buffers, dependencies);
    return task_id;
}
}  // namespace kmm
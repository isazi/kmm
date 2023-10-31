#include "kmm/runtime_impl.hpp"

#include <utility>

#include "kmm/scheduler.hpp"

namespace kmm {

BufferId RuntimeImpl::create_buffer(const BufferLayout& spec) const {
    std::lock_guard guard {m_mutex};
    BufferId id = m_dag_builder.create_buffer(spec);
    m_dag_builder.flush(*m_scheduler);
    return id;
}

void RuntimeImpl::delete_buffer(BufferId id) const {
    std::lock_guard guard {m_mutex};
    m_dag_builder.delete_buffer(id);
    m_dag_builder.flush(*m_scheduler);
}

JobId RuntimeImpl::submit_task(
    DeviceId device_id,
    std::shared_ptr<Task> task,
    std::vector<VirtualBufferRequirement> virtual_buffers,
    std::vector<JobId> dependencies) const {
    std::lock_guard guard {m_mutex};
    auto task_id = m_dag_builder.submit_task(
        device_id,
        std::move(task),
        virtual_buffers,
        std::move(dependencies));
    m_dag_builder.flush(*m_scheduler);

    return task_id;
}

JobId RuntimeImpl::submit_barrier() const {
    std::lock_guard guard {m_mutex};
    auto task_id = m_dag_builder.submit_barrier();
    m_dag_builder.flush(*m_scheduler);

    return task_id;
}

JobId RuntimeImpl::submit_buffer_barrier(BufferId id) const {
    std::lock_guard guard {m_mutex};
    auto task_id = m_dag_builder.submit_buffer_barrier(id);
    m_dag_builder.flush(*m_scheduler);

    return task_id;
}
}  // namespace kmm
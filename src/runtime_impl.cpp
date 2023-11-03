#include "kmm/runtime_impl.hpp"

#include <utility>

#include "kmm/scheduler.hpp"

namespace kmm {

RuntimeImpl::RuntimeImpl(
    std::vector<std::shared_ptr<Executor>> executors,
    std::shared_ptr<Memory> memory) :
    m_object_manager(std::make_shared<ObjectManager>()),
    m_scheduler(std::make_shared<Scheduler>(
        executors,
        std::make_shared<MemoryManager>(memory),
        m_object_manager)),
    m_thread(m_scheduler) {}

RuntimeImpl::~RuntimeImpl() {
    m_scheduler->shutdown();
    m_thread.join();
}

BufferId RuntimeImpl::create_buffer(const BufferLayout& spec) const {
    std::lock_guard guard {m_mutex};
    return m_dag_builder.create_buffer(spec);
}

void RuntimeImpl::delete_buffer(BufferId id) const {
    std::lock_guard guard {m_mutex};
    m_dag_builder.delete_buffer(id);
}

OperationId RuntimeImpl::submit_task(
    DeviceId device_id,
    std::shared_ptr<Task> task,
    std::vector<VirtualBufferRequirement> virtual_buffers,
    std::vector<OperationId> dependencies) const {
    std::lock_guard guard {m_mutex};
    return m_dag_builder
        .submit_task(device_id, std::move(task), virtual_buffers, std::move(dependencies));
}

OperationId RuntimeImpl::submit_barrier() const {
    std::lock_guard guard {m_mutex};
    return m_dag_builder.submit_barrier();
}

OperationId RuntimeImpl::submit_buffer_barrier(BufferId id) const {
    std::lock_guard guard {m_mutex};
    return m_dag_builder.submit_buffer_barrier(id);
}

OperationId RuntimeImpl::submit_promise(OperationId op_id, std::promise<void> promise) const {
    std::lock_guard guard {m_mutex};
    auto task_id = m_dag_builder.submit_promise(op_id, std::move(promise));

    // Flush to ensure that the command is actually pushed to the scheduler.
    m_dag_builder.flush(*m_scheduler);

    return task_id;
}
}  // namespace kmm
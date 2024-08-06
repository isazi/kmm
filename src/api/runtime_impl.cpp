#include "kmm/api/runtime_impl.hpp"
#include "kmm/internals/scheduler.hpp"

namespace kmm {

RuntimeImpl::RuntimeImpl(
    std::shared_ptr<Scheduler> scheduler,
    std::shared_ptr<MemoryManager> memory) :
    m_scheduler(scheduler),
    m_memory(memory) {
    m_devices = scheduler->collect_device_info();
}

TaskResult RuntimeImpl::enqueue_device_task(
    DeviceId device_id,
    std::shared_ptr<DeviceTask> task,
    const TaskRequirements& reqs) const {
    std::lock_guard guard {m_lock};

    small_vector<BufferId, 16> output_ids;
    std::vector<BufferRequirement> buffers;
    std::vector<std::shared_ptr<Buffer>> results;
    EventId event_id;

    for (const auto& input : reqs.inputs) {
        buffers.push_back(
            BufferRequirement {MemoryId(device_id), input->id(), false, input->epoch_event()});
    }

    for (const auto& output : reqs.outputs) {
        auto id = m_scheduler->create_buffer(output);
        output_ids.push_back(id);

        buffers.push_back(BufferRequirement {MemoryId(device_id), id, true, std::nullopt});
    }

    for (const auto& reduction : reqs.reductions) {
        KMM_TODO();
    }

    event_id = m_scheduler->insert_device_task(device_id, task, buffers);

    for (const auto& input : reqs.inputs) {
        input->m_users.push_back(event_id);
    }

    for (size_t i = 0; i < output_ids.size(); i++) {
        const auto& buffer_id = output_ids[i];
        const auto& layout = reqs.outputs[i];

        auto buffer = std::make_shared<Buffer>(
            shared_from_this(),
            buffer_id,
            MemoryId(device_id),
            event_id,
            layout);

        results.push_back(buffer);
    }

    return {event_id, std::move(results)};
}

TaskResult RuntimeImpl::enqueue_host_task(
    std::shared_ptr<HostTask> task,
    const TaskRequirements& reqs) const {
    KMM_TODO();
}

std::shared_ptr<Buffer> RuntimeImpl::enqueue_copies(
    MemoryId memory_id,
    BufferLayout layout,
    std::vector<CopySpecification> operations,
    std::vector<std::shared_ptr<Buffer>> buffers) {
    std::lock_guard guard {m_lock};
    KMM_ASSERT(operations.size() == buffers.size());

    auto dst_buffer = m_scheduler->create_buffer(layout);
    EventList events;

    for (size_t i = 0; i < operations.size(); i++) {
        const auto& src_buffer = buffers[i];
        const auto& op = operations[i];

        m_scheduler->insert_copy(
            src_buffer->m_id,
            src_buffer->m_owner,
            src_buffer->m_epoch,
            dst_buffer,
            memory_id,
            std::nullopt,
            op);
    }

    auto event_id = m_scheduler->join_events(events);
    return std::make_shared<Buffer>(shared_from_this(), dst_buffer, memory_id, event_id, layout);
}

bool RuntimeImpl::query_event(EventId event_id, std::chrono::system_clock::time_point deadline)
    const {
    std::lock_guard guard {m_lock};
    return m_scheduler->wait_until_ready(event_id, deadline);
}

EventId RuntimeImpl::insert_barrier() const {
    std::lock_guard guard {m_lock};
    return m_scheduler->insert_barrier();
}

EventId RuntimeImpl::join_events(const EventList& events) const {
    std::lock_guard guard {m_lock};
    return m_scheduler->join_events(events);
}

void RuntimeImpl::delete_buffer(BufferId id, const EventList& deps) const {
    std::lock_guard guard {m_lock};
    return m_scheduler->delete_buffer(id, deps);
}

const std::vector<DeviceInfo>& RuntimeImpl::devices() const {
    return m_devices;
}

Buffer::Buffer(
    std::shared_ptr<const RuntimeImpl> runtime,
    BufferId id,
    MemoryId owner,
    EventId epoch_event,
    BufferLayout layout) :
    m_runtime(runtime),
    m_id(id),
    m_owner(owner),
    m_epoch(epoch_event),
    m_layout(layout) {
    m_users.push_back(epoch_event);
}

Buffer::~Buffer() {
    m_runtime->delete_buffer(m_id, m_users);
}
}  // namespace kmm
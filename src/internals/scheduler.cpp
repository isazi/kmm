#include "kmm/internals/operation.hpp"
#include "kmm/internals/scheduler.hpp"
#include "kmm/utils/panic.hpp"

namespace kmm {

Scheduler::Scheduler(
    std::shared_ptr<MemoryManager> memory_manager,
    std::shared_ptr<CudaStreamManager> stream_manager) :
    m_memory(memory_manager),
    m_streams(stream_manager) {}

BufferId Scheduler::create_buffer(BufferLayout layout) {
    return m_memory->create_buffer(layout);
}

void Scheduler::delete_buffer(BufferId id, EventList deps) {
    m_memory->delete_buffer(id);
}

EventId Scheduler::insert_host_task(
    std::shared_ptr<HostTask> task,
    std::vector<BufferRequirement> reqs) {
    auto requests = std::vector<BufferRequest>();

    for (const auto& req : reqs) {
        requests.push_back(BufferRequest {
            .memory_id = req.memory_id,
            .buffer_id = req.buffer_id,
            .is_write = req.is_write,
            .dependency = find_event(req.dependency),
            .request = {}});
    }

    auto op = create_event<ExecuteHostOperation>(std::move(task), std::move(requests));
    m_active_events.push_back(op);
    return op->id();
}

EventId Scheduler::insert_device_task(
    DeviceId device_id,
    std::shared_ptr<DeviceTask> task,
    std::vector<BufferRequirement> reqs) {
    auto requests = std::vector<BufferRequest>();

    for (const auto& req : reqs) {
        requests.push_back(BufferRequest {
            .memory_id = req.memory_id,
            .buffer_id = req.buffer_id,
            .is_write = req.is_write,
            .dependency = find_event(req.dependency),
            .request = {}});
    }

    auto op = create_event<ExecuteDeviceOperation>(device_id, std::move(task), std::move(requests));
    m_devices.at(device_id).queue.push_back(op);
    return op->id();
}

EventId Scheduler::insert_copy(
    BufferId src_buffer_id,
    MemoryId src_memory_id,
    std::optional<EventId> src_dependency,
    BufferId dst_buffer_id,
    MemoryId dst_memory_id,
    std::optional<EventId> dst_dependency,
    CopySpecification operation) {
    auto src_buffer = BufferRequest {
        .memory_id = src_memory_id,
        .buffer_id = src_buffer_id,
        .is_write = false,
        .dependency = find_event(src_dependency)};

    auto dst_buffer = BufferRequest {
        .memory_id = dst_memory_id,
        .buffer_id = dst_buffer_id,
        .is_write = true,
        .dependency = find_event(dst_dependency)};

    // Host to host transfer
    if (src_memory_id.is_host() && dst_memory_id.is_host()) {
        KMM_TODO();
    }

    auto device_id =
        dst_memory_id.is_device() ? dst_memory_id.as_device() : src_memory_id.as_device();

    auto op = create_event<CopyDeviceOperation>(device_id, src_buffer, dst_buffer, operation);

    m_devices.at(device_id).queue.push_back(op);
    return op->id();
}

EventId Scheduler::insert_barrier() {
    EventList deps;

    for (auto& e : m_events) {
        deps.push_back(e.first);
    }

    return join_events(deps);
}

EventId Scheduler::join_events(const EventId* begin, const EventId* end) {
    OperationList deps;

    for (const auto* it = begin; it != end; it++) {
        auto e = m_events.find(*it);

        if (e != m_events.end() && !e->second->is_ready()) {
            deps.push_back(e->second);
        }
    }

    if (deps.size() == 1) {
        return deps[0]->id();
    }

    auto op = create_event<JoinOperation>(std::move(deps));
    m_active_events.push_back(op);
    return op->id();
}

std::vector<DeviceInfo> Scheduler::collect_device_info() {
    std::vector<DeviceInfo> results;

    for (size_t i = 0; i < m_devices.size(); i++) {
        results.push_back(DeviceInfo(DeviceId(i), m_devices[i].context));
    }

    return results;
}

bool Scheduler::is_ready(EventId event_id) const {
    return m_events.find(event_id) != m_events.end();
}

bool Scheduler::wait_until_ready(EventId event_id, std::chrono::system_clock::time_point deadline) {
    static constexpr auto POLL_INTERVAL = std::chrono::milliseconds(1);

    if (is_ready(event_id)) {
        return true;
    }

    auto next_poll = std::chrono::system_clock::now();

    while (true) {
        make_progress();

        if (is_ready(event_id)) {
            return true;
        }

        auto now = std::chrono::system_clock::now();

        if (now >= deadline) {
            return false;
        }

        next_poll = next_poll + POLL_INTERVAL;

        if (now < next_poll) {
            next_poll = now;
        } else {
            std::this_thread::sleep_until(std::min(next_poll, deadline));
        }
    }
}

bool Scheduler::wait_until_idle(std::chrono::system_clock::time_point deadline) {
    return wait_until_ready(insert_barrier(), deadline);
}

void Scheduler::make_progress() {
    m_streams->make_progress();

    for (auto i = 0; i < m_devices.size(); i++) {
        make_progress_device(DeviceId(i));
    }

    size_t new_size = 0;

    for (size_t i = 0; i < m_active_events.size(); i++) {
        const auto& op = m_active_events[i];

        if (op->poll(*this) == PollResult::Pending) {
            std::swap(m_active_events[new_size++], m_active_events[i]);
        } else {
            m_events.erase(op->id());
        }
    }

    m_active_events.resize(new_size);
}

void Scheduler::make_progress_device(DeviceId device_id) {
    auto& device = m_devices.at(device_id);

    for (size_t i = 0; i < device.active_streams.size(); i++) {
        if (device.queue.empty()) {
            break;
        }

        auto& op = device.active_streams[i];
        if (op && !op->is_ready()) {
            continue;
        }

        op = device.queue.front();
        device.queue.pop_front();

        auto stream = m_streams->stream_for_device(device_id, i);
        op->schedule_onto_stream(stream, *this);

        m_active_events.push_back(op);
    }
}

std::optional<std::shared_ptr<Operation>> Scheduler::find_event(
    std::optional<EventId> event_id) const {
    if (event_id) {
        auto it = m_events.find(*event_id);

        if (it != m_events.end()) {
            return it->second;
        }
    }
    return std::nullopt;
}

}  // namespace kmm
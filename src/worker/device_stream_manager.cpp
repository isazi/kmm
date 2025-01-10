#include <algorithm>
#include <queue>
#include <sstream>

#include "spdlog/spdlog.h"

#include "kmm/worker/device_stream_manager.hpp"

namespace kmm {

using Callback = std::pair<uint64_t, NotifyHandle>;

struct CompareCallback {
    bool operator()(const Callback& a, const Callback& b) const {
        return a.first > b.first;
    }
};

struct DeviceStreamManager::StreamState {
    KMM_NOT_COPYABLE(StreamState)

  public:
    StreamState(size_t pool_index, GPUContextHandle c, GPUstream_t s) :
        pool_index(pool_index),
        context(c),
        gpu_stream(s) {}
    StreamState(StreamState&&) = default;

    size_t pool_index;
    GPUContextHandle context;
    GPUstream_t gpu_stream;
    std::deque<GPUevent_t> pending_events;
    uint64_t first_pending_index = 1;
    std::priority_queue<Callback, std::vector<Callback>, CompareCallback> callbacks_heap;
};

struct DeviceStreamManager::EventPool {
    EventPool(GPUContextHandle context) : m_context(context) {}
    EventPool(EventPool&&) noexcept = default;
    EventPool(const EventPool&) = delete;
    ~EventPool();
    GPUevent_t pop();
    void push(GPUevent_t event);

    GPUContextHandle m_context;
    std::vector<GPUevent_t> m_events;
};

DeviceStreamManager::DeviceStreamManager() {}

DeviceStream DeviceStreamManager::create_stream(GPUContextHandle context, bool high_priority) {
    size_t pool_index;
    bool found_pool = false;

    for (size_t i = 0; i < m_event_pools.size(); i++) {
        if (m_event_pools[i].m_context == context) {
            found_pool = true;
            pool_index = i;
        }
    }

    if (!found_pool) {
        pool_index = m_event_pools.size();
        m_event_pools.push_back(EventPool(context));
    }

    GPUContextGuard guard {context};

    int least_priority;
    int greatest_priority;
    KMM_GPU_CHECK(gpuCtxGetStreamPriorityRange(&least_priority, &greatest_priority));
    int priority = high_priority ? greatest_priority : least_priority;

    size_t index = m_streams.size();
    GPUstream_t gpu_stream;
    KMM_GPU_CHECK(gpuStreamCreateWithPriority(&gpu_stream, GPU_STREAM_NON_BLOCKING, priority));
    m_streams.emplace_back(pool_index, context, gpu_stream);

    return DeviceStream(checked_cast<uint8_t>(index));
}

DeviceStreamManager::~DeviceStreamManager() {
    for (auto& stream : m_streams) {
        GPUContextGuard guard {stream.context};

        KMM_GPU_CHECK(gpuStreamSynchronize(stream.gpu_stream));
        KMM_ASSERT(gpuStreamQuery(stream.gpu_stream) == GPU_SUCCESS);
        KMM_GPU_CHECK(gpuStreamDestroy(stream.gpu_stream));

        for (const auto& gpu_event : stream.pending_events) {
            KMM_GPU_CHECK(gpuEventSynchronize(gpu_event));
            KMM_ASSERT(gpuEventSynchronize(gpu_event) == GPU_SUCCESS);

            stream.first_pending_index += 1;
            m_event_pools[stream.pool_index].push(gpu_event);
        }
    }
}

void DeviceStreamManager::wait_until_idle() const {
    for (const auto& stream : m_streams) {
        KMM_GPU_CHECK(gpuStreamSynchronize(stream.gpu_stream));
    }
}

void DeviceStreamManager::wait_until_ready(DeviceStream stream) const {
    KMM_GPU_CHECK(gpuStreamSynchronize(get(stream)));
}

void DeviceStreamManager::wait_until_ready(DeviceEvent event) const {
    KMM_ASSERT(event.stream() < m_streams.size());
    const auto& src_stream = m_streams[event.stream()];

    if (event.index() < src_stream.first_pending_index) {
        return;
    }

    auto offset = event.index() - src_stream.first_pending_index;
    GPUevent_t gpu_event = src_stream.pending_events.at(offset);

    GPUContextGuard guard {src_stream.context};
    KMM_GPU_CHECK(gpuEventSynchronize(gpu_event));
}

void DeviceStreamManager::wait_until_ready(const DeviceEventSet& events) const {
    for (DeviceEvent e : events) {
        wait_until_ready(e);
    }
}

bool DeviceStreamManager::is_idle() const {
    for (const auto& stream : m_streams) {
        if (!stream.pending_events.empty()) {
            return false;
        }

        if (!stream.callbacks_heap.empty()) {
            return false;
        }

        GPUContextGuard guard {stream.context};
        KMM_GPU_CHECK(gpuStreamSynchronize(stream.gpu_stream));
        KMM_GPU_CHECK(gpuStreamSynchronize(nullptr));
    }

    return true;
}

bool DeviceStreamManager::is_ready(DeviceStream stream) const noexcept {
    KMM_ASSERT(stream < m_streams.size());
    return m_streams[stream].pending_events.empty();
}

bool DeviceStreamManager::is_ready(DeviceEvent event) const noexcept {
    KMM_ASSERT(event.stream() < m_streams.size());
    return m_streams[event.stream()].first_pending_index > event.index();
}

bool DeviceStreamManager::is_ready(const DeviceEventSet& events) const noexcept {
    for (DeviceEvent e : events) {
        if (!is_ready(e)) {
            return false;
        }
    }

    return true;
}

bool DeviceStreamManager::is_ready(DeviceEventSet& events) const noexcept {
    return events.remove_ready_trailing(*this);
}

void DeviceStreamManager::attach_callback(DeviceEvent event, NotifyHandle callback) {
    KMM_ASSERT(event.stream() < m_streams.size());
    auto& stream = m_streams[event.stream()];
    stream.callbacks_heap.emplace(event.index(), std::move(callback));
}

void DeviceStreamManager::attach_callback(DeviceStream stream, NotifyHandle callback) {
    attach_callback(record_event(stream), std::move(callback));
}

DeviceEvent DeviceStreamManager::record_event(DeviceStream stream_id) {
    KMM_ASSERT(stream_id < m_streams.size());
    auto& stream = m_streams[stream_id];

    uint64_t event_index = stream.first_pending_index + stream.pending_events.size();
    auto event = DeviceEvent {stream_id, event_index};

    GPUevent_t gpu_event = m_event_pools[stream.pool_index].pop();
    stream.pending_events.push_back(gpu_event);

    KMM_GPU_CHECK(gpuEventRecord(gpu_event, stream.gpu_stream));

    spdlog::trace("GPU stream {} records new GPU event {}", stream_id, event);
    return event;
}

void DeviceStreamManager::wait_on_default_stream(DeviceStream stream_id) {
    KMM_ASSERT(stream_id < m_streams.size());
    auto& stream = m_streams[stream_id];

    GPUevent_t gpu_event = m_event_pools[stream.pool_index].pop();
    m_event_pools[stream.pool_index].push(gpu_event);

    KMM_GPU_CHECK(gpuEventRecord(gpu_event, 0));
    KMM_GPU_CHECK(gpuStreamWaitEvent(stream.gpu_stream, gpu_event, GPU_EVENT_WAIT_DEFAULT));
}

void DeviceStreamManager::wait_for_event(DeviceStream stream, DeviceEvent event) const {
    KMM_ASSERT(event.stream() < m_streams.size());
    KMM_ASSERT(stream < m_streams.size());

    // Stream never needs to wait on events from itself
    if (event.stream() == stream) {
        return;
    }

    const auto& src_stream = m_streams.at(event.stream());
    const auto& dst_stream = m_streams.at(stream);

    // Event has already completed, no need to wait.
    if (event.index() < src_stream.first_pending_index) {
        return;
    }

    auto offset = event.index() - src_stream.first_pending_index;
    GPUevent_t gpu_event = src_stream.pending_events.at(offset);
    KMM_GPU_CHECK(gpuStreamWaitEvent(dst_stream.gpu_stream, gpu_event, GPU_EVENT_WAIT_DEFAULT));

    spdlog::trace("GPU stream {} must wait on GPU event {}", stream, event);
}

void DeviceStreamManager::wait_for_events(
    DeviceStream stream,
    const DeviceEvent* begin,
    const DeviceEvent* end
) const {
    for (const auto* it = begin; it != end; it++) {
        wait_for_event(stream, *it);
    }
}

void DeviceStreamManager::wait_for_events(DeviceStream stream, const DeviceEventSet& events) const {
    wait_for_events(stream, events.begin(), events.end());
}

void DeviceStreamManager::wait_for_events(
    DeviceStream stream,
    const std::vector<DeviceEvent>& events
) const {
    wait_for_events(stream, &*events.begin(), &*events.end());
}

bool DeviceStreamManager::event_happens_before(DeviceEvent source, DeviceEvent target) const {
    return source.stream() == target.stream() && source.index() < target.index();
}

GPUContextHandle DeviceStreamManager::context(DeviceStream stream) const {
    KMM_ASSERT(stream.get() < m_streams.size());
    return m_streams[stream.get()].context;
}

GPUstream_t DeviceStreamManager::get(DeviceStream stream) const {
    KMM_ASSERT(stream < m_streams.size());
    return m_streams[stream].gpu_stream;
}

bool DeviceStreamManager::make_progress() {
    bool update_happened = false;

    for (size_t i = 0; i < m_streams.size(); i++) {
        auto& stream = m_streams[i];

        if (!stream.pending_events.empty()) {
            GPUContextGuard guard {stream.context};

            do {
                GPUevent_t gpu_event = stream.pending_events[0];
                GPUresult result = gpuEventQuery(gpu_event);

                if (result == GPU_ERROR_NOT_READY) {
                    break;
                }

                if (result != GPU_SUCCESS) {
                    throw GPUDriverException("`gpuEventQuery` failed", result);
                }

                spdlog::trace(
                    "GPU event {} completed",
                    DeviceEvent(static_cast<uint8_t>(i), stream.first_pending_index)
                );

                stream.first_pending_index += 1;
                stream.pending_events.pop_front();
                m_event_pools[stream.pool_index].push(gpu_event);
                update_happened = true;
            } while (!stream.pending_events.empty());
        }

        while (!stream.callbacks_heap.empty()) {
            const auto& [index, handle] = stream.callbacks_heap.top();

            if (index >= stream.first_pending_index) {
                break;
            }

            handle.notify();
            update_happened = true;

            stream.callbacks_heap.pop();
        }
    }

    return update_happened;
}

DeviceStreamManager::EventPool::~EventPool() {
    GPUContextGuard guard {m_context};

    for (const auto& gpu_event : m_events) {
        KMM_GPU_CHECK(gpuEventDestroy(gpu_event));
    }
}

GPUevent_t DeviceStreamManager::EventPool::pop() {
    GPUevent_t gpu_event;

    if (m_events.empty()) {
        GPUContextGuard guard {m_context};
        KMM_GPU_CHECK(gpuEventCreate(&gpu_event, GPU_EVENT_DISABLE_TIMING));
    } else {
        gpu_event = m_events.back();
        m_events.pop_back();
    }

    return gpu_event;
}

void DeviceStreamManager::EventPool::push(GPUevent_t event) {
    m_events.push_back(event);
}

DeviceEventSet::DeviceEventSet(DeviceEvent e) {
    m_events.push_back(e);
}

DeviceEventSet::DeviceEventSet(std::initializer_list<DeviceEvent> e) {
    m_events.insert_all(e.begin(), e.end());
}

DeviceEventSet& DeviceEventSet::operator=(std::initializer_list<DeviceEvent> e) {
    clear();
    m_events.insert_all(e.begin(), e.end());
    return *this;
}

void DeviceEventSet::insert(DeviceEvent e) noexcept {
    static constexpr size_t INVALID_INDEX = std::numeric_limits<size_t>::max();
    size_t found_index = INVALID_INDEX;

    if (e.is_null()) {
        return;
    }

    for (size_t i = 0; i < m_events.size(); i++) {
        if (m_events[i].stream() == e.stream()) {
            found_index = i;
        }
    }

    if (found_index != INVALID_INDEX) {
        m_events[found_index] = std::max(m_events[found_index], e);
    } else {
        KMM_ASSERT(m_events.try_push_back(e));
    }
}

void DeviceEventSet::insert(const DeviceEventSet& that) noexcept {
    static constexpr size_t INVALID_INDEX = std::numeric_limits<size_t>::max();
    size_t num_old_events = m_events.size();

    for (auto e : that.m_events) {
        size_t found_index = INVALID_INDEX;

        for (size_t i = 0; i < num_old_events; i++) {
            if (m_events[i].stream() == e.stream()) {
                found_index = i;
            }
        }

        if (found_index != INVALID_INDEX) {
            m_events[found_index] = std::max(m_events[found_index], e);
        } else {
            KMM_ASSERT(m_events.try_push_back(e));
        }
    }
}

void DeviceEventSet::insert(DeviceEventSet&& that) noexcept {
    if (m_events.is_empty()) {
        m_events = std::move(that.m_events);
        return;
    }

    insert(that);
}

bool DeviceEventSet::remove_ready(const DeviceStreamManager& m) noexcept {
    size_t old_size = m_events.size();
    size_t new_size = 0;

    for (size_t index = 0; index < old_size; index++) {
        auto event = m_events[index];

        if (!m.is_ready(event)) {
            m_events[new_size] = event;
            new_size++;
        }
    }

    m_events.truncate(new_size);
    return new_size > 0;
}

bool DeviceEventSet::remove_ready_trailing(const DeviceStreamManager& m) noexcept {
    for (size_t new_size = m_events.size(); new_size > 0; new_size--) {
        auto event = m_events[new_size - 1];

        // event is not ready, truncate events up to `new_size`.
        if (!m.is_ready(event)) {
            m_events.truncate(new_size);
            return false;
        }
    }

    // all events are ready
    m_events.clear();
    return true;
}

void DeviceEventSet::clear() noexcept {
    m_events.clear();
}

bool DeviceEventSet::is_empty() const noexcept {
    return m_events.is_empty();
}

const DeviceEvent* DeviceEventSet::begin() const noexcept {
    return m_events.begin();
}

const DeviceEvent* DeviceEventSet::end() const noexcept {
    return m_events.end();
}

DeviceEventSet operator|(const DeviceEventSet& a, const DeviceEventSet& b) noexcept {
    DeviceEventSet result = a;
    result.insert(b);
    return result;
}

std::ostream& operator<<(std::ostream& f, const DeviceStream& e) {
    return f << uint32_t(e.get());
}

std::ostream& operator<<(std::ostream& f, const DeviceEvent& e) {
    if (e.m_event_and_stream == 0) {
        return f << "<none>";
    }

    return f << e.stream() << ":" << e.index();
}

std::ostream& operator<<(std::ostream& f, const DeviceEventSet& events) {
    // Sort events
    auto sorted_events = std::vector<DeviceEvent> {events.begin(), events.end()};
    std::sort(sorted_events.begin(), sorted_events.end());

    // Remove duplicates
    auto it = std::unique(sorted_events.begin(), sorted_events.end());
    sorted_events.erase(it, sorted_events.end());

    bool is_first = true;
    f << "[";

    for (auto e : sorted_events) {
        // Skip empty events
        if (e.is_null()) {
            continue;
        }

        if (!is_first) {
            f << ", ";
        }

        is_first = false;
        f << e;
    }

    f << "]";
    return f;
}

}  // namespace kmm
#include <algorithm>
#include <queue>
#include <sstream>

#include "spdlog/spdlog.h"

#include "kmm/internals/cuda_stream_manager.hpp"

namespace kmm {

using Callback = std::pair<uint64_t, NotifyHandle>;

struct CompareCallback {
    bool operator()(const Callback& a, const Callback& b) const {
        return a.first > b.first;
    }
};

struct CudaStreamManager::StreamState {
    KMM_NOT_COPYABLE(StreamState)

  public:
    StreamState(size_t pool_index, CudaContextHandle c, CUstream s) :
        pool_index(pool_index),
        context(c),
        cuda_stream(s) {}
    StreamState(StreamState&&) = default;

    size_t pool_index;
    CudaContextHandle context;
    CUstream cuda_stream;
    std::deque<CUevent> pending_events;
    uint64_t first_pending_index = 1;
    std::priority_queue<Callback, std::vector<Callback>, CompareCallback> callbacks_heap;
};

struct CudaStreamManager::EventPool {
    EventPool(CudaContextHandle context) : m_context(context) {}
    EventPool(EventPool&&) noexcept = default;
    EventPool(const EventPool&) = delete;
    ~EventPool();
    CUevent pop();
    void push(CUevent event);

    CudaContextHandle m_context;
    std::vector<CUevent> m_events;
};

CudaStreamManager::CudaStreamManager() {}

DeviceStream CudaStreamManager::create_stream(CudaContextHandle context, bool high_priority) {
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

    CudaContextGuard guard {context};

    int least_priority;
    int greatest_priority;
    KMM_CUDA_CHECK(cuCtxGetStreamPriorityRange(&least_priority, &greatest_priority));
    int priority = high_priority ? greatest_priority : least_priority;

    size_t index = m_streams.size();
    CUstream cuda_stream;
    KMM_CUDA_CHECK(cuStreamCreateWithPriority(&cuda_stream, CU_STREAM_NON_BLOCKING, priority));
    m_streams.emplace_back(pool_index, context, cuda_stream);

    return DeviceStream(index);
}

CudaStreamManager::~CudaStreamManager() {
    for (auto& stream : m_streams) {
        CudaContextGuard guard {stream.context};

        KMM_CUDA_CHECK(cuStreamSynchronize(stream.cuda_stream));
        KMM_ASSERT(cuStreamQuery(stream.cuda_stream) == CUDA_SUCCESS);
        KMM_CUDA_CHECK(cuStreamDestroy(stream.cuda_stream));

        for (const auto& cuda_event : stream.pending_events) {
            KMM_CUDA_CHECK(cuEventSynchronize(cuda_event));
            KMM_ASSERT(cuEventSynchronize(cuda_event) == CUDA_SUCCESS);

            stream.first_pending_index += 1;
            m_event_pools[stream.pool_index].push(cuda_event);
        }
    }
}

void CudaStreamManager::wait_until_idle() const {
    for (const auto& stream : m_streams) {
        KMM_CUDA_CHECK(cuStreamSynchronize(stream.cuda_stream));
    }
}

void CudaStreamManager::wait_until_ready(DeviceStream stream) const {
    KMM_CUDA_CHECK(cuStreamSynchronize(get(stream)));
}

void CudaStreamManager::wait_until_ready(DeviceEvent event) const {
    const auto& src_stream = m_streams[event.stream()];

    if (event.index() < src_stream.first_pending_index) {
        return;
    }

    auto offset = event.index() - src_stream.first_pending_index;
    CUevent cuda_event = src_stream.pending_events.at(offset);

    CudaContextGuard guard {src_stream.context};
    KMM_CUDA_CHECK(cuEventSynchronize(cuda_event));
}

void CudaStreamManager::wait_until_ready(const DeviceEventSet& events) const {
    for (DeviceEvent e : events) {
        wait_until_ready(e);
    }
}

bool CudaStreamManager::is_idle() const {
    for (const auto& stream : m_streams) {
        if (!stream.pending_events.empty() || !stream.callbacks_heap.empty()) {
            return false;
        }

        CudaContextGuard guard {stream.context};
        KMM_CUDA_CHECK(cuStreamSynchronize(stream.cuda_stream));
        KMM_CUDA_CHECK(cuStreamSynchronize(nullptr));
    }

    return true;
}

bool CudaStreamManager::is_ready(DeviceStream stream) const {
    return m_streams.at(stream).pending_events.empty();
}

bool CudaStreamManager::is_ready(DeviceEvent event) const {
    return m_streams.at(event.stream()).first_pending_index > event.index();
}

bool CudaStreamManager::is_ready(const DeviceEventSet& events) const {
    for (DeviceEvent e : events) {
        if (!is_ready(e)) {
            return false;
        }
    }

    return true;
}

bool CudaStreamManager::is_ready(DeviceEventSet& events) const {
    events.remove_completed(*this);
    return events.begin() == events.end();
}

void CudaStreamManager::attach_callback(DeviceEvent event, NotifyHandle callback) {
    auto& stream = m_streams[event.stream()];
    stream.callbacks_heap.emplace(event.index(), std::move(callback));
}

void CudaStreamManager::attach_callback(DeviceStream stream, NotifyHandle callback) {
    attach_callback(record_event(stream), std::move(callback));
}

DeviceEvent CudaStreamManager::record_event(DeviceStream stream_id) {
    auto& stream = m_streams.at(stream_id);

    uint64_t event_index = stream.first_pending_index + stream.pending_events.size();
    auto event = DeviceEvent {stream_id, event_index};

    CUevent cuda_event = m_event_pools[stream.pool_index].pop();
    stream.pending_events.push_back(cuda_event);

    KMM_CUDA_CHECK(cuEventRecord(cuda_event, stream.cuda_stream));

    spdlog::trace("CUDA stream {} records new CUDA event {}", stream_id, event);
    return event;
}

void CudaStreamManager::wait_on_default_stream(DeviceStream stream_id) {
    auto& stream = m_streams.at(stream_id);

    CUevent cuda_event = m_event_pools[stream.pool_index].pop();
    m_event_pools[stream.pool_index].push(cuda_event);

    KMM_CUDA_CHECK(cuEventRecord(cuda_event, 0));
    KMM_CUDA_CHECK(cuStreamWaitEvent(stream.cuda_stream, cuda_event, CU_EVENT_WAIT_DEFAULT));
}

void CudaStreamManager::wait_for_event(DeviceStream stream, DeviceEvent event) const {
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
    CUevent cuda_event = src_stream.pending_events.at(offset);
    KMM_CUDA_CHECK(cuStreamWaitEvent(dst_stream.cuda_stream, cuda_event, CU_EVENT_WAIT_DEFAULT));

    spdlog::trace("CUDA stream {} must wait on CUDA event {}", stream, event);
}

void CudaStreamManager::wait_for_events(
    DeviceStream stream,
    const DeviceEvent* begin,
    const DeviceEvent* end
) {
    for (const auto* it = begin; it != end; it++) {
        wait_for_event(stream, *it);
    }
}

void CudaStreamManager::wait_for_events(DeviceStream stream, const DeviceEventSet& events) {
    wait_for_events(stream, events.begin(), events.end());
}

void CudaStreamManager::wait_for_events(
    DeviceStream stream,
    const std::vector<DeviceEvent>& events
) {
    wait_for_events(stream, &*events.begin(), &*events.end());
}

bool CudaStreamManager::event_happens_before(DeviceEvent source, DeviceEvent target) const {
    return source.stream() == target.stream() && source.index() < target.index();
}

CudaContextHandle CudaStreamManager::context(DeviceStream stream) const {
    KMM_ASSERT(stream.get() < m_streams.size());
    return m_streams[stream.get()].context;
}

CUstream CudaStreamManager::get(DeviceStream stream) const {
    KMM_ASSERT(stream < m_streams.size());
    return m_streams[stream].cuda_stream;
}

bool CudaStreamManager::make_progress() {
    bool update_happened = false;

    for (size_t i = 0; i < m_streams.size(); i++) {
        auto& stream = m_streams[i];

        if (!stream.pending_events.empty()) {
            CudaContextGuard guard {stream.context};

            do {
                CUevent cuda_event = stream.pending_events[0];
                CUresult result = cuEventQuery(cuda_event);

                if (result == CUDA_ERROR_NOT_READY) {
                    break;
                }

                if (result != CUDA_SUCCESS) {
                    throw CudaDriverException("`cuEventQuery` failed", result);
                }

                spdlog::trace(
                    "CUDA event {} completed",
                    DeviceEvent(i, stream.first_pending_index)
                );

                stream.first_pending_index += 1;
                stream.pending_events.pop_front();
                m_event_pools[stream.pool_index].push(cuda_event);
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

CudaStreamManager::EventPool::~EventPool() {
    CudaContextGuard guard {m_context};

    for (const auto& cuda_event : m_events) {
        KMM_CUDA_CHECK(cuEventDestroy(cuda_event));
    }
}

CUevent CudaStreamManager::EventPool::pop() {
    CUevent cuda_event;

    if (m_events.empty()) {
        CudaContextGuard guard {m_context};
        KMM_CUDA_CHECK(cuEventCreate(&cuda_event, CU_EVENT_DISABLE_TIMING));
    } else {
        cuda_event = m_events.back();
        m_events.pop_back();
    }

    return cuda_event;
}

void CudaStreamManager::EventPool::push(CUevent event) {
    m_events.push_back(event);
}

std::ostream& operator<<(std::ostream& f, const DeviceStream& e) {
    return f << uint32_t(e.get());
}

std::ostream& operator<<(std::ostream& f, const DeviceEvent& e) {
    return f << e.stream() << ":" << e.index();
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

void DeviceEventSet::insert(DeviceEvent e) {
    bool found = false;
    size_t found_index;

    for (size_t i = 0; i < m_events.size(); i++) {
        if (m_events[i].stream() == e.stream()) {
            found = true;
            found_index = i;
        }
    }

    if (found) {
        m_events[found_index] = std::max(m_events[found_index], e);
    } else {
        m_events.push_back(e);
    }
}

void DeviceEventSet::insert(const DeviceEventSet& events) {
    size_t n = m_events.size();

    for (DeviceEvent e : events) {
        bool found = false;
        size_t found_index;

        for (size_t i = 0; i < n; i++) {
            if (m_events[i].stream() == e.stream()) {
                found = true;
                found_index = i;
            }
        }

        if (found) {
            m_events[found_index] = std::max(m_events[found_index], e);
        } else {
            m_events.push_back(e);
        }
    }
}

void DeviceEventSet::insert(DeviceEventSet&& events) {
    if (m_events.is_empty()) {
        m_events = std::move(events.m_events);
    } else {
        insert(events);
    }
}

void DeviceEventSet::remove_completed(const CudaStreamManager& m) {
    size_t index = 0;
    size_t new_size = m_events.size();

    while (index < new_size) {
        if (m.is_ready(m_events[index])) {
            m_events[index] = m_events[new_size - 1];
            new_size--;
        } else {
            index++;
        }
    }

    m_events.resize(new_size);
}

void DeviceEventSet::clear() {
    m_events.clear();
}

bool DeviceEventSet::is_empty() const {
    return m_events.is_empty();
}

const DeviceEvent* DeviceEventSet::begin() const {
    return m_events.begin();
}

const DeviceEvent* DeviceEventSet::end() const {
    return m_events.end();
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
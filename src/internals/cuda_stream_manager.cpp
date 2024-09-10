#include <algorithm>
#include <sstream>

#include "spdlog/spdlog.h"

#include "kmm/internals/cuda_stream_manager.hpp"

namespace kmm {

std::ostream& operator<<(std::ostream& f, const CudaStream& e) {
    return f << uint32_t(e.get());
}

std::ostream& operator<<(std::ostream& f, const CudaEvent& e) {
    return f << e.stream() << ":" << e.event();
}

CudaEventSet::CudaEventSet(CudaEvent e) {
    m_events.push_back(e);
}

CudaEventSet::CudaEventSet(std::initializer_list<CudaEvent> e) {
    m_events.insert_all(e.begin(), e.end());
}

CudaEventSet& CudaEventSet::operator=(std::initializer_list<CudaEvent> e) {
    clear();
    m_events.insert_all(e.begin(), e.end());
    return *this;
}

void CudaEventSet::insert(CudaEvent e) {
    for (CudaEvent& p : m_events) {
        if (p.stream() == e.stream()) {
            p = std::max(e, p);
            return;
        }
    }

    m_events.push_back(e);
}

void CudaEventSet::insert(const CudaEventSet& events) {
    for (CudaEvent e : events) {
        insert(e);
    }
}

void CudaEventSet::remove_completed(const CudaStreamManager& m) {
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

void CudaEventSet::clear() {
    m_events.clear();
}

const CudaEvent* CudaEventSet::begin() const {
    return m_events.begin();
}

const CudaEvent* CudaEventSet::end() const {
    return m_events.end();
}

std::ostream& operator<<(std::ostream& f, const CudaEventSet& events) {
    // Sort events
    auto sorted_events = std::vector<CudaEvent> {events.begin(), events.end()};
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

struct CudaStreamManager::StreamState {
    KMM_NOT_COPYABLE(StreamState)

  public:
    StreamState(DeviceId device_id, CudaContextHandle c, CUstream s) :
        device_id(device_id),
        context(c),
        cuda_stream(s) {}
    StreamState(StreamState&&) = default;

    DeviceId device_id;
    CudaContextHandle context;
    CUstream cuda_stream;
    std::deque<CUevent> pending_events;
    uint64_t first_pending_index = 1;
    std::vector<std::pair<uint64_t, NotifyHandle>> callbacks_heap;
};

struct CudaStreamManager::EventPool {
    EventPool(CudaContextHandle context) : m_context(context) {}
    ~EventPool();
    CUevent pop();
    void push(CUevent event);

    CudaContextHandle m_context;
    std::vector<CUevent> m_events;
};

CudaStreamManager::CudaStreamManager(
    const std::vector<CudaContextHandle>& contexts,
    size_t streams_per_device) :
    m_streams_per_device(streams_per_device) {
    for (size_t i = 0; i < contexts.size(); i++) {
        const auto& context = contexts[i];
        m_event_pools.emplace_back(context);

        for (size_t j = 0; j < streams_per_device; j++) {
            CudaContextGuard guard {context};

            CUstream cuda_stream;
            KMM_CUDA_CHECK(cuStreamCreate(&cuda_stream, CU_STREAM_NON_BLOCKING));
            m_streams.emplace_back(DeviceId(i), context, cuda_stream);
        }
    }
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
            m_event_pools[stream.device_id].push(cuda_event);
        }
    }
}

CudaStream CudaStreamManager::stream_for_device(DeviceId device_id, size_t stream_index) const {
    return {checked_cast<uint8_t>(device_id.get() + (stream_index % m_streams_per_device))};
}

DeviceId CudaStreamManager::device_from_stream(CudaStream stream) const {
    KMM_ASSERT(stream < m_streams.size());
    return m_streams[stream].device_id;
}

void CudaStreamManager::wait_until_idle() const {
    for (const auto& stream : m_streams) {
        KMM_CUDA_CHECK(cuStreamSynchronize(stream.cuda_stream));
    }
}

void CudaStreamManager::wait_until_ready(CudaStream stream) const {
    KMM_CUDA_CHECK(cuStreamSynchronize(get(stream)));
}

void CudaStreamManager::wait_until_ready(CudaEvent event) const {
    auto device_id = device_from_stream(event.stream());
    const auto& src_stream = m_streams[event.stream()];

    if (event.event() < src_stream.first_pending_index) {
        return;
    }

    auto offset = event.event() - src_stream.first_pending_index;
    CUevent cuda_event = src_stream.pending_events.at(offset);

    CudaContextGuard guard {m_event_pools[device_id].m_context};
    KMM_CUDA_CHECK(cuEventSynchronize(cuda_event));
}

void CudaStreamManager::wait_until_ready(const CudaEventSet& events) const {
    for (CudaEvent e : events) {
        wait_until_ready(e);
    }
}

bool CudaStreamManager::is_ready(CudaStream stream) const {
    return m_streams.at(stream).pending_events.empty();
}

bool CudaStreamManager::is_ready(CudaEvent event) const {
    return m_streams.at(event.stream()).first_pending_index > event.event();
}

bool CudaStreamManager::is_ready(const CudaEventSet& events) const {
    for (CudaEvent e : events) {
        if (!is_ready(e)) {
            return false;
        }
    }

    return true;
}

bool CudaStreamManager::is_ready(CudaEventSet& events) const {
    events.remove_completed(*this);
    return events.begin() == events.end();
}

void CudaStreamManager::attach_callback(CudaEvent event, NotifyHandle callback) {
    auto& stream = m_streams[event.stream()];

    stream.callbacks_heap.emplace_back(event.event(), std::move(callback));

    std::push_heap(
        stream.callbacks_heap.begin(),
        stream.callbacks_heap.end(),
        [](auto& a, auto& b) { return a.first > b.first; });
}

void CudaStreamManager::attach_callback(CudaStream stream, NotifyHandle callback) {
    attach_callback(record_event(stream), std::move(callback));
}

CudaEvent CudaStreamManager::record_event(CudaStream stream_id) {
    DeviceId device_id = device_from_stream(stream_id);
    auto& stream = m_streams.at(stream_id);
    CUevent event = m_event_pools[device_id].pop();

    uint64_t event_index = stream.first_pending_index + stream.pending_events.size();
    stream.pending_events.push_back(event);

    KMM_CUDA_CHECK(cuEventRecord(event, stream.cuda_stream));

    return CudaEvent {stream_id, event_index};
}

void CudaStreamManager::wait_on_default_stream(CudaStream stream_id) {
    auto& stream = m_streams.at(stream_id);

    CUevent cuda_event = m_event_pools[stream.device_id].pop();
    m_event_pools[stream.device_id].push(cuda_event);

    KMM_CUDA_CHECK(cuEventRecord(cuda_event, 0));
    KMM_CUDA_CHECK(cuStreamWaitEvent(stream.cuda_stream, cuda_event, CU_EVENT_WAIT_DEFAULT));
}

void CudaStreamManager::wait_for_event(CudaStream stream, CudaEvent event) const {
    const auto& src_stream = m_streams.at(event.stream());
    const auto& dst_stream = m_streams.at(stream);

    if (event.event() < src_stream.first_pending_index) {
        return;
    }

    auto offset = event.event() - src_stream.first_pending_index;
    CUevent cuda_event = src_stream.pending_events.at(offset);

    KMM_CUDA_CHECK(cuStreamWaitEvent(dst_stream.cuda_stream, cuda_event, CU_EVENT_WAIT_DEFAULT));
}

void CudaStreamManager::wait_for_events(
    CudaStream stream,
    const CudaEvent* begin,
    const CudaEvent* end) {
    std::vector<CudaEvent> events = {begin, end};
    std::sort(events.begin(), events.end());
    CudaEventSet deps;
    for (auto e : events) {
        deps.insert(e);
    }

    spdlog::warn("stream {} waits for events: {}", stream.get(), deps);

    for (const auto* it = begin; it != end; it++) {
        wait_for_event(stream, *it);
    }
}

void CudaStreamManager::wait_for_events(CudaStream stream, const CudaEventSet& events) {
    wait_for_events(stream, events.begin(), events.end());
}

void CudaStreamManager::wait_for_events(CudaStream stream, const std::vector<CudaEvent>& events) {
    wait_for_events(stream, &*events.begin(), &*events.end());
}

bool CudaStreamManager::event_happens_before(CudaEvent source, CudaEvent target) const {
    return source.stream() == target.stream() && source.event() <= target.event();
}

CudaContextHandle CudaStreamManager::get(DeviceId id) const {
    KMM_ASSERT(id < m_event_pools.size());
    return m_event_pools[id].m_context;
}

CUstream CudaStreamManager::get(CudaStream stream) const {
    KMM_ASSERT(stream < m_streams.size());
    return m_streams[stream].cuda_stream;
}

bool CudaStreamManager::make_progress() {
    bool update_happened = false;

    for (auto& stream : m_streams) {
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

                stream.first_pending_index += 1;
                stream.pending_events.pop_front();
                m_event_pools[stream.device_id].push(cuda_event);
                update_happened = true;
            } while (!stream.pending_events.empty());
        }

        while (!stream.callbacks_heap.empty()) {
            auto& [index, handle] = stream.callbacks_heap.front();

            if (index >= stream.first_pending_index) {
                break;
            }

            handle.notify();
            update_happened = true;

            std::pop_heap(
                stream.callbacks_heap.begin(),
                stream.callbacks_heap.end(),
                [](auto& a, auto& b) { return a.first > b.first; });

            stream.callbacks_heap.pop_back();
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

}  // namespace kmm
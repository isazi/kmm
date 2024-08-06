#include "kmm/internals/cuda_stream_manager.hpp"
#include "kmm/utils/panic.hpp"

namespace kmm {

CudaStreamManager::CudaStreamManager(
    std::vector<CudaContextHandle> contexts,
    size_t nstreams_per_device) {
    m_streams_per_device = nstreams_per_device;

    for (const auto& context : contexts) {
        CudaContextGuard guard {context};

        for (size_t i = 0; i < nstreams_per_device; i++) {
            CUstream cuda_stream;
            KMM_CUDA_CHECK(cuStreamCreate(&cuda_stream, CU_STREAM_NON_BLOCKING));

            m_streams.emplace_back(context, cuda_stream);
        }
    }
}

CudaStreamManager::~CudaStreamManager() {
    wait_until_idle();

    for (const auto& stream : m_streams) {
        KMM_CUDA_CHECK(cuStreamDestroy(stream.cuda_stream));

        for (const auto& event : stream.pending_events) {
            KMM_CUDA_CHECK(cuEventDestroy(event));
        }
    }

    for (const auto& event : m_event_pool) {
        KMM_CUDA_CHECK(cuEventDestroy(event));
    }
}

CudaStreamId CudaStreamManager::stream_for_device(DeviceId device_id, size_t stream_index) const {
    return {
        uint8_t(device_id.get() * m_streams_per_device + (stream_index % m_streams_per_device))};
}

DeviceId CudaStreamManager::device_from_stream(CudaStreamId stream) const {
    return DeviceId(stream.index / m_streams_per_device);
}

void CudaStreamManager::wait_until_idle() const {
    for (const auto& stream : m_streams) {
        KMM_CUDA_CHECK(cuStreamSynchronize(stream.cuda_stream));

        for (const auto& event : stream.pending_events) {
            KMM_CUDA_CHECK(cuEventSynchronize(event));
        }
    }

    for (const auto& event : m_event_pool) {
        KMM_CUDA_CHECK(cuEventSynchronize(event));
    }
}

bool CudaStreamManager::is_ready(CudaEvent event) const {
    const auto& stream = m_streams.at(event.stream.index);
    return event.index < stream.first_pending_index;
}

CudaEvent CudaStreamManager::record_event(CudaStreamId stream) {
    auto& target_stream = m_streams.at(stream.index);
    uint64_t event_index = target_stream.first_pending_index
        + static_cast<uint64_t>(target_stream.pending_events.size());

    CUevent cuda_event = pop_event();
    KMM_CUDA_CHECK(cuEventRecord(cuda_event, target_stream.cuda_stream));

    target_stream.pending_events.push_back(cuda_event);
    m_event_pool.pop_back();

    return CudaEvent {stream, event_index};
}
CUevent CudaStreamManager::pop_event() {
    if (m_event_pool.empty()) {
        CUevent cuda_event;
        KMM_CUDA_CHECK(cuEventCreate(&cuda_event, CU_EVENT_DISABLE_TIMING));
        m_event_pool.push_back(cuda_event);
    }

    CUevent cuda_event = m_event_pool.back();
    m_event_pool.pop_back();

    return cuda_event;
}

void CudaStreamManager::wait_on_default_stream(CudaStreamId stream) {
    CUstream cuda_stream = m_streams[stream.index].cuda_stream;
    CUevent event = pop_event();

    KMM_CUDA_CHECK(cuEventRecord(event, 0));
    KMM_CUDA_CHECK(cuStreamWaitEvent(cuda_stream, event, CU_EVENT_WAIT_DEFAULT));

    m_event_pool.push_back(event);
}

void CudaStreamManager::wait_for_event(CudaStreamId stream, CudaEvent event) const {
    // There is no need for a stream to wait for itself
    if (stream.index == event.stream.index) {
        return;
    }

    const auto& source_stream = m_streams.at(event.stream.index);

    // Event has already finished, no need to wait for it
    if (event.index < source_stream.first_pending_index) {
        return;
    }

    size_t offset = static_cast<size_t>(event.index - source_stream.first_pending_index);
    CUevent cuda_event = source_stream.pending_events.at(offset);

    const auto& target_stream = m_streams.at(stream.index);
    KMM_CUDA_CHECK(cuStreamWaitEvent(target_stream.cuda_stream, cuda_event, CU_EVENT_WAIT_DEFAULT));
}

void CudaStreamManager::wait_for_events(CudaStreamId stream, const CudaEventSet& events) {
    events.wait_for_all(*this, stream);
}

void CudaStreamManager::wait_for_events(
    CudaStreamId stream,
    const CudaEvent* begin,
    const CudaEvent* end) {
    for (const auto* it = begin; it != end; it++) {
        wait_for_event(stream, *it);
    }
}

void CudaStreamManager::wait_for_events(CudaStreamId stream, const std::vector<CudaEvent>& events) {
    return wait_for_events(stream, events.data(), events.data() + events.size());
}

bool CudaStreamManager::event_happens_before(CudaEvent source, CudaEvent target) const {
    return source.stream == target.stream && source.index <= target.index;
}

CUstream CudaStreamManager::get(CudaStreamId stream) const {
    return m_streams.at(stream.index).cuda_stream;
}

void CudaStreamManager::make_progress() {
    for (auto& stream : m_streams) {
        while (!stream.pending_events.empty()) {
            CUevent cuda_event = stream.pending_events.front();
            CUresult result = cuEventQuery(cuda_event);

            // Event not ready, break from while loop
            if (result == CUresult::CUDA_ERROR_NOT_READY) {
                break;
            }

            // Unexpected error, throw exception
            if (result != CUresult::CUDA_SUCCESS) {
                throw CudaDriverException("error occurred in `cuEventQuery`", result);
            }

            // Event completed, increment `first_pending_index` counter and move event to pool
            stream.first_pending_index++;
            stream.pending_events.pop_front();
            m_event_pool.push_back(cuda_event);
        }
    }
}

void CudaEventSet::insert(CudaStreamManager& manager, CudaEvent new_event) {
    if (manager.is_ready(new_event)) {
        return;
    }

    for (auto& e : m_events) {
        if (manager.event_happens_before(e, new_event) || manager.is_ready(e)) {
            e = new_event;
            return;
        }
    }

    m_events.push_back(new_event);
}

void CudaEventSet::clear() {
    m_events.clear();
}

void CudaEventSet::wait_for_all(CudaStreamManager& manager, CudaStreamId stream) const {
    for (const auto& event : m_events) {
        manager.wait_for_event(stream, event);
    }
}

}  // namespace kmm
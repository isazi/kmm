#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>

#include "kmm/core/identifiers.hpp"
#include "kmm/utils/cuda.hpp"
#include "kmm/utils/small_vector.hpp"

namespace kmm {

class CudaStreamManager;

struct CudaStreamId {
    uint8_t index = 0;
};

inline bool operator==(const CudaStreamId& a, CudaStreamId& b) {
    return a.index == b.index;
}

struct CudaEvent {
    CudaStreamId stream;
    uint64_t index = 0;
};

inline bool operator==(const CudaEvent& a, CudaEvent& b) {
    return a.stream == b.stream && a.index == b.index;
}

class CudaEventSet {
  public:
    void clear();
    void insert(CudaStreamManager& manager, CudaEvent new_event);
    void wait_for_all(CudaStreamManager& manager, CudaStreamId stream) const;

  private:
    small_vector<CudaEvent, 2> m_events;
};

class CudaStreamManager {
    struct StreamBookkeeping {
        KMM_NOT_COPYABLE(StreamBookkeeping)

      public:
        StreamBookkeeping(CudaContextHandle c, CUstream s) : context(c), cuda_stream(s) {}
        StreamBookkeeping(StreamBookkeeping&&) = default;

        CudaContextHandle context;
        CUstream cuda_stream;
        std::deque<CUevent> pending_events;
        uint64_t first_pending_index = 1;
    };

  public:
    CudaStreamManager(std::vector<CudaContextHandle> contexts, size_t nstreams_per_device);
    ~CudaStreamManager();

    CudaStreamId stream_for_device(DeviceId device_id, size_t stream_index = 0) const;
    DeviceId device_from_stream(CudaStreamId stream) const;

    void wait_until_idle() const;
    bool is_ready(CudaEvent event) const;
    CudaEvent record_event(CudaStreamId stream);
    void wait_on_default_stream(CudaStreamId stream);
    void wait_for_event(CudaStreamId stream, CudaEvent event) const;

    void wait_for_events(CudaStreamId stream, const CudaEventSet& events);
    void wait_for_events(CudaStreamId stream, const CudaEvent* begin, const CudaEvent* end);
    void wait_for_events(CudaStreamId stream, const std::vector<CudaEvent>& events);

    /**
     * Check if the given `source` event must occur before the given `target` event. In other words,
     * if this function returns true, then `source` must be triggered before `target` can trigger.
     */
    bool event_happens_before(CudaEvent source, CudaEvent target) const;

    CUstream get(CudaStreamId stream) const;

    void make_progress();

  private:
    CUevent pop_event();

    std::vector<StreamBookkeeping> m_streams;
    std::vector<CUevent> m_event_pool;
    size_t m_streams_per_device;
};

}  // namespace kmm
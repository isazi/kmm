#pragma once

#include <deque>
#include <map>

#include "kmm/core/identifiers.hpp"
#include "kmm/utils/cuda.hpp"
#include "kmm/utils/notify.hpp"
#include "kmm/utils/small_vector.hpp"

namespace kmm {

class CudaStreamManager;

struct CudaStream {
    uint8_t stream_index = 0;
};

struct CudaEvent {
    CudaStream stream;
    uint64_t event_index = 0;
};

class CudaEventSet {
  public:
    CudaEventSet() = default;
    CudaEventSet(const CudaEventSet&) = default;
    CudaEventSet(CudaEvent);
    CudaEventSet(std::initializer_list<CudaEvent>);

    void insert(CudaStreamManager& manager, CudaEvent new_event);
    void insert(CudaStreamManager& manager, CudaEventSet& new_events);
    void prune(CudaStreamManager& manager);
    void clear();
    const CudaEvent* begin() const;
    const CudaEvent* end() const;

    CudaEventSet& operator=(const CudaEventSet&) = default;
    CudaEventSet& operator=(std::initializer_list<CudaEvent>);

  private:
    small_vector<CudaEvent, 2> m_events;
};

class CudaStreamManager {
    KMM_NOT_COPYABLE_OR_MOVABLE(CudaStreamManager)

  public:
    CudaStreamManager(const std::vector<CudaContextHandle>& contexts, size_t streams_per_device);
    ~CudaStreamManager();

    CudaStream stream_for_device(DeviceId device_id, size_t stream_index = 0) const;
    DeviceId device_from_stream(CudaStream stream) const;

    void wait_until_idle() const;

    void wait_until_ready(CudaStream stream) const;
    void wait_until_ready(CudaEvent event) const;
    void wait_until_ready(CudaEventSet events) const;

    bool is_ready(CudaStream stream) const;
    bool is_ready(CudaEvent event) const;
    bool is_ready(CudaEventSet events) const;

    void attach_callback(CudaEvent event, NotifyHandle callback);
    void attach_callback(CudaStream event, NotifyHandle callback);

    CudaEvent record_event(CudaStream stream);
    void wait_on_default_stream(CudaStream stream);

    void wait_for_event(CudaStream stream, CudaEvent event) const;
    void wait_for_events(CudaStream stream, const CudaEventSet& events);
    void wait_for_events(CudaStream stream, const CudaEvent* begin, const CudaEvent* end);
    void wait_for_events(CudaStream stream, const std::vector<CudaEvent>& events);

    /**
     * Check if the given `source` event must occur before the given `target` event. In other words,
     * if this function returns true, then `source` must be triggered before `target` can trigger.
     */
    bool event_happens_before(CudaEvent source, CudaEvent target) const;

    CudaContextHandle get(DeviceId device_id) const;
    CUstream get(CudaStream stream) const;

    template<typename F>
    CudaEvent with_stream(CudaStream stream, CudaEventSet deps, F fun) {
        wait_for_events(stream, deps);

        try {
            fun(get(stream));
        } catch (...) {
            wait_until_ready(stream);
            throw;
        }

        return record_event(stream);
    }

    template<typename F>
    CudaEvent with_stream(CudaStream stream, F fun) {
        return with_stream(stream, {}, fun);
    }

    bool make_progress();

  private:
    struct StreamState;
    struct EventPool;

    std::vector<StreamState> m_streams;
    std::vector<EventPool> m_event_pools;
    size_t m_streams_per_device;
};

}  // namespace kmm
#pragma once

#include <deque>
#include <map>

#include "kmm/core/identifiers.hpp"
#include "kmm/utils/cuda.hpp"
#include "kmm/utils/notify.hpp"
#include "kmm/utils/small_vector.hpp"

namespace kmm {

class CudaStreamManager;

class CudaStream {
  public:
    CudaStream(uint8_t i = 0) : m_index(i) {}

    uint8_t get() const {
        return m_index;
    }

    operator uint8_t() const {
        return m_index;
    }

  private:
    uint8_t m_index;
};

class CudaEvent {
  public:
    CudaEvent() = default;

    CudaEvent(CudaStream stream, uint64_t event) {
        KMM_ASSERT(event < (1ULL << 56));
        m_event_and_stream = (uint64_t(stream.get()) << 56) | event;
    }

    CudaStream stream() const {
        return static_cast<uint8_t>(m_event_and_stream >> 56);
    }

    uint64_t event() const {
        return m_event_and_stream & uint64_t(0x00FFFFFFFFFFFFFF);
    }

    constexpr bool operator==(const CudaEvent& that) const {
        return that.m_event_and_stream == m_event_and_stream;
    }

    constexpr bool operator<(const CudaEvent& that) const {
        // This is equivalent to tuple(this.stream, this.event) < tuple(that.stream, that.event)
        return that.m_event_and_stream < m_event_and_stream;
    }

    KMM_IMPL_COMPARISON_OPS(CudaEvent)

  private:
    uint64_t m_event_and_stream = 0;
};

class CudaEventSet {
  public:
    CudaEventSet() = default;
    CudaEventSet(const CudaEventSet&) = default;
    CudaEventSet(CudaEventSet&&) noexcept = default;

    CudaEventSet(CudaEvent);
    CudaEventSet(std::initializer_list<CudaEvent>);

    CudaEventSet& operator=(const CudaEventSet&) = default;
    CudaEventSet& operator=(CudaEventSet&&) noexcept = default;
    CudaEventSet& operator=(std::initializer_list<CudaEvent>);

    void insert(CudaEvent e);
    void insert(const CudaEventSet& e);
    void remove_completed(const CudaStreamManager&);
    void clear();

    const CudaEvent* begin() const;
    const CudaEvent* end() const;

  private:
    small_vector<CudaEvent, 4> m_events;
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
    void wait_until_ready(const CudaEventSet& events) const;

    bool is_ready(CudaStream stream) const;
    bool is_ready(CudaEvent event) const;
    bool is_ready(const CudaEventSet& events) const;
    bool is_ready(CudaEventSet& events) const;

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
    CudaEvent with_stream(CudaStream stream, const CudaEventSet& deps, F fun) {
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
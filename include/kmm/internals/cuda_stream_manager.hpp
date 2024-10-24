#pragma once

#include <deque>
#include <map>

#include "kmm/core/identifiers.hpp"
#include "kmm/utils/cuda.hpp"
#include "kmm/utils/notify.hpp"
#include "kmm/utils/small_vector.hpp"

namespace kmm {

class CudaStreamManager;
class DeviceStream;
class DeviceEvent;
class DeviceEventSet;

class CudaStreamManager {
    KMM_NOT_COPYABLE_OR_MOVABLE(CudaStreamManager)

  public:
    CudaStreamManager();
    ~CudaStreamManager();

    bool make_progress();

    DeviceStream create_stream(CudaContextHandle context, bool high_priority = false);

    void wait_until_idle() const;
    void wait_until_ready(DeviceStream stream) const;
    void wait_until_ready(DeviceEvent event) const;
    void wait_until_ready(const DeviceEventSet& events) const;

    bool is_idle() const;
    bool is_ready(DeviceStream stream) const;
    bool is_ready(DeviceEvent event) const;
    bool is_ready(const DeviceEventSet& events) const;
    bool is_ready(DeviceEventSet& events) const;

    void attach_callback(DeviceEvent event, NotifyHandle callback);
    void attach_callback(DeviceStream event, NotifyHandle callback);

    DeviceEvent record_event(DeviceStream stream);
    void wait_on_default_stream(DeviceStream stream);

    void wait_for_event(DeviceStream stream, DeviceEvent event) const;
    void wait_for_events(DeviceStream stream, const DeviceEventSet& events);
    void wait_for_events(DeviceStream stream, const DeviceEvent* begin, const DeviceEvent* end);
    void wait_for_events(DeviceStream stream, const std::vector<DeviceEvent>& events);

    /**
     * Check if the given `source` event must occur before the given `target` event. In other words,
     * if this function returns true, then `source` must be triggered before `target` can trigger.
     */
    bool event_happens_before(DeviceEvent source, DeviceEvent target) const;

    CudaContextHandle context(DeviceStream device_id) const;
    CUstream get(DeviceStream stream) const;

    template<typename F>
    DeviceEvent with_stream(DeviceStream stream, const DeviceEventSet& deps, F fun);

    template<typename F>
    DeviceEvent with_stream(DeviceStream stream, F fun);

  private:
    struct StreamState;
    struct EventPool;

    std::vector<StreamState> m_streams;
    std::vector<EventPool> m_event_pools;
};

class DeviceStream {
  public:
    DeviceStream(uint8_t i = 0) : m_index(i) {}

    uint8_t get() const {
        return m_index;
    }

    operator uint8_t() const {
        return m_index;
    }

    friend std::ostream& operator<<(std::ostream&, const DeviceStream& e);

  private:
    uint8_t m_index;
};

class DeviceEvent {
  public:
    DeviceEvent() = default;

    DeviceEvent(DeviceStream stream, uint64_t index) {
        KMM_ASSERT(index < (1ULL << 56));
        m_event_and_stream = (uint64_t(stream.get()) << 56) | index;
    }

    DeviceStream stream() const {
        return static_cast<uint8_t>(m_event_and_stream >> 56);
    }

    uint64_t index() const {
        return m_event_and_stream & uint64_t(0x00FFFFFFFFFFFFFF);
    }

    constexpr bool operator==(const DeviceEvent& that) const {
        return this->m_event_and_stream == that.m_event_and_stream;
    }

    constexpr bool operator<(const DeviceEvent& that) const {
        // This is equivalent to tuple(this.stream, this.event) < tuple(that.stream, that.event)
        return this->m_event_and_stream < that.m_event_and_stream;
    }

    KMM_IMPL_COMPARISON_OPS(DeviceEvent)

    friend std::ostream& operator<<(std::ostream&, const DeviceEvent& e);

  private:
    uint64_t m_event_and_stream = 0;
};

class DeviceEventSet {
  public:
    DeviceEventSet() = default;
    DeviceEventSet(const DeviceEventSet&) = default;
    DeviceEventSet(DeviceEventSet&&) noexcept = default;

    DeviceEventSet(DeviceEvent);
    DeviceEventSet(std::initializer_list<DeviceEvent>);

    DeviceEventSet& operator=(const DeviceEventSet&) = default;
    DeviceEventSet& operator=(DeviceEventSet&&) noexcept = default;
    DeviceEventSet& operator=(std::initializer_list<DeviceEvent>);

    void insert(DeviceEvent e);
    void insert(const DeviceEventSet& e);
    void insert(DeviceEventSet&& e);
    void remove_completed(const CudaStreamManager&);
    void clear();

    bool is_empty() const;
    const DeviceEvent* begin() const;
    const DeviceEvent* end() const;

    friend std::ostream& operator<<(std::ostream&, const DeviceEventSet& e);

  private:
    small_vector<DeviceEvent, 2> m_events;
};

template<typename F>
DeviceEvent CudaStreamManager::with_stream(DeviceStream stream, const DeviceEventSet& deps, F fun) {
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
DeviceEvent CudaStreamManager::with_stream(DeviceStream stream, F fun) {
    return with_stream(stream, {}, fun);
}

}  // namespace kmm

template<>
struct fmt::formatter<kmm::DeviceStream>: fmt::ostream_formatter {};
template<>
struct fmt::formatter<kmm::DeviceEvent>: fmt::ostream_formatter {};
template<>
struct fmt::formatter<kmm::DeviceEventSet>: fmt::ostream_formatter {};
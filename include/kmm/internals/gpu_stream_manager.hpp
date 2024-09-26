#pragma once

#include <deque>
#include <map>

#include "kmm/core/identifiers.hpp"
#include "kmm/utils/gpu.hpp"
#include "kmm/utils/notify.hpp"
#include "kmm/utils/small_vector.hpp"

namespace kmm {

class GPUStreamManager;
class GPUStream;
class GPUEvent;
class GPUEventSet;

class GPUStreamManager {
    KMM_NOT_COPYABLE_OR_MOVABLE(GPUStreamManager)

  public:
    GPUStreamManager();
    ~GPUStreamManager();

    bool make_progress();

    GPUStream create_stream(GPUContextHandle context, bool high_priority = false);

    void wait_until_idle() const;

    void wait_until_ready(GPUStream stream) const;
    void wait_until_ready(GPUEvent event) const;
    void wait_until_ready(const GPUEventSet& events) const;

    bool is_ready(GPUStream stream) const;
    bool is_ready(GPUEvent event) const;
    bool is_ready(const GPUEventSet& events) const;
    bool is_ready(GPUEventSet& events) const;

    void attach_callback(GPUEvent event, NotifyHandle callback);
    void attach_callback(GPUStream event, NotifyHandle callback);

    GPUEvent record_event(GPUStream stream);
    void wait_on_default_stream(GPUStream stream);

    void wait_for_event(GPUStream stream, GPUEvent event) const;
    void wait_for_events(GPUStream stream, const GPUEventSet& events);
    void wait_for_events(GPUStream stream, const GPUEvent* begin, const GPUEvent* end);
    void wait_for_events(GPUStream stream, const std::vector<GPUEvent>& events);

    /**
     * Check if the given `source` event must occur before the given `target` event. In other words,
     * if this function returns true, then `source` must be triggered before `target` can trigger.
     */
    bool event_happens_before(GPUEvent source, GPUEvent target) const;

    GPUContextHandle context(GPUStream device_id) const;
    stream_t get(GPUStream stream) const;

    template<typename F>
    GPUEvent with_stream(GPUStream stream, const GPUEventSet& deps, F fun);

    template<typename F>
    GPUEvent with_stream(GPUStream stream, F fun);

  private:
    struct StreamState;
    struct EventPool;

    std::vector<StreamState> m_streams;
    std::vector<EventPool> m_event_pools;
};

class GPUStream {
  public:
    GPUStream(uint8_t i = 0) : m_index(i) {}

    uint8_t get() const {
        return m_index;
    }

    operator uint8_t() const {
        return m_index;
    }

    friend std::ostream& operator<<(std::ostream&, const GPUStream& e);

  private:
    uint8_t m_index;
};

class GPUEvent {
  public:
    GPUEvent() = default;

    GPUEvent(GPUStream stream, uint64_t index) {
        KMM_ASSERT(index < (1ULL << 56));
        m_event_and_stream = (uint64_t(stream.get()) << 56) | index;
    }

    GPUStream stream() const {
        return static_cast<uint8_t>(m_event_and_stream >> 56);
    }

    uint64_t index() const {
        return m_event_and_stream & uint64_t(0x00FFFFFFFFFFFFFF);
    }

    constexpr bool operator==(const GPUEvent& that) const {
        return that.m_event_and_stream == m_event_and_stream;
    }

    constexpr bool operator<(const GPUEvent& that) const {
        // This is equivalent to tuple(this.stream, this.event) < tuple(that.stream, that.event)
        return that.m_event_and_stream < m_event_and_stream;
    }

    KMM_IMPL_COMPARISON_OPS(GPUEvent)

    friend std::ostream& operator<<(std::ostream&, const GPUEvent& e);

  private:
    uint64_t m_event_and_stream = 0;
};

class GPUEventSet {
  public:
    GPUEventSet() = default;
    GPUEventSet(const GPUEventSet&) = default;
    GPUEventSet(GPUEventSet&&) noexcept = default;

    GPUEventSet(GPUEvent);
    GPUEventSet(std::initializer_list<GPUEvent>);

    GPUEventSet& operator=(const GPUEventSet&) = default;
    GPUEventSet& operator=(GPUEventSet&&) noexcept = default;
    GPUEventSet& operator=(std::initializer_list<GPUEvent>);

    void insert(GPUEvent e);
    void insert(const GPUEventSet& e);
    void insert(GPUEventSet&& e);
    void remove_completed(const GPUStreamManager&);
    void clear();

    const GPUEvent* begin() const;
    const GPUEvent* end() const;

    friend std::ostream& operator<<(std::ostream&, const GPUEventSet& e);

  private:
    small_vector<GPUEvent, 4> m_events;
};

template<typename F>
GPUEvent GPUStreamManager::with_stream(GPUStream stream, const GPUEventSet& deps, F fun) {
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
GPUEvent GPUStreamManager::with_stream(GPUStream stream, F fun) {
    return with_stream(stream, {}, fun);
}

}  // namespace kmm

template<>
struct fmt::formatter<kmm::GPUStream>: fmt::ostream_formatter {};
template<>
struct fmt::formatter<kmm::GPUEvent>: fmt::ostream_formatter {};
template<>
struct fmt::formatter<kmm::GPUEventSet>: fmt::ostream_formatter {};
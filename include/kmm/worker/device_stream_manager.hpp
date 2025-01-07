#pragma once

#include <deque>
#include <map>

#include "kmm/core/identifiers.hpp"
#include "kmm/utils/gpu.hpp"
#include "kmm/utils/notify.hpp"
#include "kmm/utils/small_vector.hpp"

namespace kmm {

class DeviceStreamManager;
class DeviceStream;
class DeviceEvent;
class DeviceEventSet;

class DeviceStreamManager {
    KMM_NOT_COPYABLE_OR_MOVABLE(DeviceStreamManager)

  public:
    DeviceStreamManager();
    ~DeviceStreamManager();

    bool make_progress();

    DeviceStream create_stream(GPUContextHandle context, bool high_priority = false);

    void wait_until_idle() const;
    void wait_until_ready(DeviceStream stream) const;
    void wait_until_ready(DeviceEvent event) const;
    void wait_until_ready(const DeviceEventSet& events) const;

    bool is_idle() const;
    bool is_ready(DeviceStream stream) const noexcept;
    bool is_ready(DeviceEvent event) const noexcept;
    bool is_ready(const DeviceEventSet& events) const noexcept;
    bool is_ready(DeviceEventSet& events) const noexcept;

    void attach_callback(DeviceEvent event, NotifyHandle callback);
    void attach_callback(DeviceStream stream, NotifyHandle callback);

    DeviceEvent record_event(DeviceStream stream);
    void wait_on_default_stream(DeviceStream stream);

    void wait_for_event(DeviceStream stream, DeviceEvent event) const;
    void wait_for_events(DeviceStream stream, const DeviceEventSet& events) const;
    void wait_for_events(DeviceStream stream, const DeviceEvent* begin, const DeviceEvent* end)
        const;
    void wait_for_events(DeviceStream stream, const std::vector<DeviceEvent>& events) const;

    /**
     * Check if the given `source` event must occur before the given `target` event. In other words,
     * if this function returns true, then `source` must be triggered before `target` can trigger.
     */
    bool event_happens_before(DeviceEvent source, DeviceEvent target) const;

    GPUContextHandle context(DeviceStream stream) const;
    GPUstream_t get(DeviceStream stream) const;

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

    DeviceEvent(DeviceStream stream, uint64_t index) noexcept {
        KMM_ASSERT(index < (1ULL << 56));
        m_event_and_stream = (uint64_t(stream.get()) << 56) | index;
    }

    bool is_null() const noexcept {
        return m_event_and_stream == 0;
    }

    DeviceStream stream() const noexcept {
        return static_cast<uint8_t>(m_event_and_stream >> 56);
    }

    uint64_t index() const noexcept {
        return m_event_and_stream & uint64_t(0x00FFFFFFFFFFFFFF);
    }

    constexpr bool operator==(const DeviceEvent& that) const noexcept {
        return this->m_event_and_stream == that.m_event_and_stream;
    }

    constexpr bool operator<(const DeviceEvent& that) const noexcept {
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
    DeviceEventSet(std::initializer_list<DeviceEvent>);
    DeviceEventSet(DeviceEvent);

    DeviceEventSet& operator=(const DeviceEventSet&) = default;
    DeviceEventSet& operator=(DeviceEventSet&&) noexcept = default;
    DeviceEventSet& operator=(std::initializer_list<DeviceEvent>);

    /**
     * Insert the given event into the set.
     */
    void insert(DeviceEvent e) noexcept;

    /**
     * Insert all events from `that` into this set.
     */
    void insert(const DeviceEventSet& that) noexcept;

    /**
     * Insert all events from `that` into this set.
     */
    void insert(DeviceEventSet&& that) noexcept;

    /**
     * Remove all events from the list for which the manager indicates that they are ready.
     *
     * @return true if all events were ready, false otherwise.
     */
    bool remove_ready(const DeviceStreamManager&) noexcept;

    /**
     * Remove events from the list for which the manager indicates that they are ready. This
     * differs from `remove_ready` in that it only remove sthe events at the end of the list and
     * does not reorder the leading events in the list.
     *
     * @return true if all events were ready, false otherwise.
     */
    bool remove_ready_trailing(const DeviceStreamManager&) noexcept;

    /**
     * Remove all events.
     */
    void clear() noexcept;

    /**
     * Returns `true` if the set is empty, `false` otherwise.
     */
    bool is_empty() const noexcept;

    /**
     * Returns pointer to the first event.
     */
    const DeviceEvent* begin() const noexcept;

    /**
     * Returns pointer to one past the last event.
     */
    const DeviceEvent* end() const noexcept;

    friend DeviceEventSet operator|(const DeviceEventSet& a, const DeviceEventSet& b) noexcept;
    friend std::ostream& operator<<(std::ostream&, const DeviceEventSet& e);

  private:
    small_vector<DeviceEvent, 2> m_events;
};

template<typename F>
DeviceEvent DeviceStreamManager::with_stream(
    DeviceStream stream,
    const DeviceEventSet& deps,
    F fun
) {
    wait_for_events(stream, deps);
    return with_stream(stream, std::move(fun));
}

template<typename F>
DeviceEvent DeviceStreamManager::with_stream(DeviceStream stream, F fun) {
    try {
        fun(get(stream));
        return record_event(stream);
    } catch (...) {
        wait_until_ready(stream);
        throw;
    }
}

}  // namespace kmm

template<>
struct fmt::formatter<kmm::DeviceStream>: fmt::ostream_formatter {};
template<>
struct fmt::formatter<kmm::DeviceEvent>: fmt::ostream_formatter {};
template<>
struct fmt::formatter<kmm::DeviceEventSet>: fmt::ostream_formatter {};
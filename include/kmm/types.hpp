#pragma once

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "fmt/format.h"

// Macro for implementing comparison operators for class `T`
#define KMM_IMPL_COMPARISON_OPS(T)         \
    bool operator!=(const T& that) const { \
        return !(*this == that);           \
    }                                      \
    bool operator<=(const T& that) const { \
        return !(*this > that);            \
    }                                      \
    bool operator>(const T& that) const {  \
        return that < *this;               \
    }                                      \
    bool operator>=(const T& that) const { \
        return that <= *this;              \
    }

namespace kmm {

class RuntimeImpl;
class Runtime;
class BlockHeader;

template<typename T, typename Tag = void>
class Identifier {
  public:
    explicit constexpr Identifier(T value) : m_value(value) {}

    static constexpr Identifier invalid() {
        return Identifier(std::numeric_limits<T>::max());
    }

    T get() const {
        return m_value;
    }

    operator T() const {
        return get();
    }

    bool operator==(const Identifier& that) const {
        return m_value == that.m_value;
    }

    bool operator<(const Identifier& that) const {
        return m_value < that.m_value;
    }

    KMM_IMPL_COMPARISON_OPS(Identifier)

  private:
    T m_value;
};

template<typename T, typename Tag>
T format_as(const Identifier<T, Tag>& id) {
    return id.get();
}

using index_t = int;
using DeviceId = Identifier<uint8_t, struct DeviceTag>;
using EventId = Identifier<uint64_t, struct EventTag>;

class BlockId {
  public:
    explicit constexpr BlockId(EventId event, uint8_t index) : m_event(event), m_index(index) {}

    static constexpr BlockId invalid() {
        return BlockId(EventId::invalid(), 0);
    }

    EventId event() const {
        return m_event;
    }

    size_t hash() const {
        return m_event.get() * 256 * m_index;
    }

    std::string to_string() const {
        return std::to_string(m_event.get()) + ":" + std::to_string(m_index);
    }

    bool operator==(const BlockId& that) const {
        return m_event == that.m_event && m_index == that.m_index;
    }

    bool operator<(const BlockId& that) const {
        return m_event < that.m_event || (m_event == that.m_event && m_index < that.m_index);
    }

    KMM_IMPL_COMPARISON_OPS(BlockId)

  private:
    EventId m_event;
    uint8_t m_index;
};

inline std::string format_as(const BlockId& id) {
    return id.to_string();
}

class BufferId {
  public:
    explicit constexpr BufferId(uint64_t index, size_t num_bytes) :
        m_index(index),
        m_nbytes(num_bytes) {}

    static constexpr BufferId invalid() {
        return BufferId(std::numeric_limits<uint64_t>::max(), 0);
    }

    uint64_t get() const {
        return m_index;
    }

    operator uint64_t() const {
        return m_index;
    }

    size_t num_bytes() const {
        return m_nbytes;
    }

    size_t hash() const {
        return m_index;
    }

    bool operator==(const BufferId& that) const {
        return m_index == that.m_index && m_nbytes == that.m_nbytes;
    }

    bool operator<(const BufferId& that) const {
        return m_index < that.m_index || (m_index == that.m_index && m_nbytes < that.m_nbytes);
    }

    KMM_IMPL_COMPARISON_OPS(BufferId)

  private:
    uint64_t m_index;
    size_t m_nbytes;
};

inline uint64_t format_as(const BufferId& id) {
    return id.get();
}

class EventList {
  public:
    EventList() = default;
    EventList(std::initializer_list<EventId> list) {
        extend(list.begin(), list.size());
    }
    EventList(const std::vector<EventId>& list) {
        extend(list.data(), list.size());
    }

    const EventId* begin() const {
        return &*m_events.begin();
    }

    const EventId* end() const {
        return begin() + size();
    }

    EventId operator[](size_t index) const {
        return *(begin() + index);
    }

    size_t size() const {
        return m_events.size();
    }

    void extend(const EventId* data, size_t len) {
        m_events.insert(m_events.end(), data, data + len);
    }

    void extend(const EventList& that) {
        extend(that.begin(), that.size());
    }

    void push_back(EventId event) {
        m_events.push_back(event);
    }

    void remove_duplicates() {
        std::sort(m_events.begin(), m_events.end());
        auto last_unique = std::unique(std::begin(m_events), std::end(m_events));
        m_events.erase(last_unique, std::end(m_events));
    }

  private:
    std::vector<EventId> m_events = {};
};

enum class PollResult { Pending, Ready };

class Waker {
  public:
    virtual ~Waker() = default;
    virtual void trigger_wakeup(bool allow_progress = false) const = 0;
};

using WakerRef = const std::enable_shared_from_this<Waker>&;

}  // namespace kmm

namespace std {
template<>
struct hash<kmm::DeviceId>: hash<uint8_t> {};

template<>
struct hash<kmm::EventId>: hash<uint64_t> {};

template<>
struct hash<kmm::BufferId>: hash<uint64_t> {};

template<>
struct hash<kmm::BlockId> {
    size_t operator()(kmm::BlockId b) const {
        return b.hash();
    }
};
}  // namespace std
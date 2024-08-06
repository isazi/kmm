#pragma once

#include "kmm/utils/checked_math.hpp"
#include "kmm/utils/panic.hpp"
#include "kmm/utils/small_vector.hpp"

#define KMM_IMPL_COMPARISON_OPS(T)                   \
    constexpr bool operator!=(const T& that) const { \
        return !(*this == that);                     \
    }                                                \
    constexpr bool operator<=(const T& that) const { \
        return !(*this > that);                      \
    }                                                \
    constexpr bool operator>(const T& that) const {  \
        return that < *this;                         \
    }                                                \
    constexpr bool operator>=(const T& that) const { \
        return that <= *this;                        \
    }

namespace kmm {

using index_t = int;

struct NodeId {
    explicit constexpr NodeId(uint8_t v) : m_value(v) {}
    explicit constexpr NodeId(size_t v) : m_value(checked_cast<uint8_t>(v)) {}

    constexpr uint8_t get() const {
        return m_value;
    }

    operator uint8_t() const {
        return get();
    }

    constexpr bool operator==(const NodeId& that) const {
        return m_value == that.m_value;
    }

    constexpr bool operator<(const NodeId& that) const {
        return m_value < that.m_value;
    }

    KMM_IMPL_COMPARISON_OPS(NodeId)

  private:
    uint8_t m_value;
};

struct DeviceId {
    explicit constexpr DeviceId(uint8_t v) : m_value(v) {}

    template<typename T>
    explicit constexpr DeviceId(T v) : m_value(checked_cast<uint8_t>(v)) {}

    constexpr uint8_t get() const {
        return m_value;
    }

    operator uint8_t() const {
        return get();
    }

    constexpr bool operator==(const DeviceId& that) const {
        return m_value == that.m_value;
    }

    constexpr bool operator<(const DeviceId& that) const {
        return m_value < that.m_value;
    }

    KMM_IMPL_COMPARISON_OPS(DeviceId)

  private:
    uint8_t m_value;
};

struct MemoryId {
  private:
    constexpr MemoryId(uint8_t v) : m_value(v) {}

  public:
    explicit constexpr MemoryId(DeviceId device) : MemoryId(device.get()) {}

    static constexpr MemoryId host() {
        return MemoryId {HOST_ID};
    }

    bool is_host() const {
        return m_value == HOST_ID;
    }

    bool is_device() const {
        return m_value != HOST_ID;
    }

    DeviceId as_device() const {
        KMM_ASSERT(is_device());
        return DeviceId(m_value);
    }

    friend bool operator==(MemoryId lhs, MemoryId rhs) {
        return lhs.m_value == rhs.m_value;
    }

    friend bool operator==(MemoryId lhs, DeviceId rhs) {
        return lhs == MemoryId(rhs);
    }

    friend bool operator==(DeviceId lhs, MemoryId rhs) {
        return MemoryId(lhs) == rhs.m_value;
    }

    friend bool operator!=(MemoryId lhs, MemoryId rhs) {
        return !(lhs == rhs);
    }

    friend bool operator!=(DeviceId lhs, MemoryId rhs) {
        return !(lhs == rhs);
    }

    friend bool operator!=(MemoryId lhs, DeviceId rhs) {
        return !(lhs == rhs);
    }

  private:
    static constexpr uint8_t HOST_ID = 0xff;
    uint8_t m_value = HOST_ID;
};

struct BufferId {
    explicit constexpr BufferId(uint64_t v = ~0) : m_value(v) {}

    constexpr uint64_t get() const {
        return m_value;
    }

    operator uint64_t() const {
        return get();
    }

    constexpr bool operator==(const BufferId& that) const {
        return m_value == that.m_value;
    }

    constexpr bool operator<(const BufferId& that) const {
        return m_value < that.m_value;
    }

    KMM_IMPL_COMPARISON_OPS(BufferId)

  private:
    uint64_t m_value;
};

struct EventId {
    explicit constexpr EventId(uint64_t v = 0) : m_value(v) {}

    constexpr uint64_t get() const {
        return m_value;
    }

    operator uint64_t() const {
        return get();
    }

    constexpr bool operator==(const EventId& that) const {
        return m_value == that.m_value;
    }

    constexpr bool operator<(const EventId& that) const {
        return m_value < that.m_value;
    }

    KMM_IMPL_COMPARISON_OPS(EventId)

  private:
    uint64_t m_value;
};

using EventList = small_vector<EventId, 2>;

}  // namespace kmm

namespace std {

template<>
struct hash<kmm::NodeId>: hash<uint8_t> {};

template<>
struct hash<kmm::DeviceId>: hash<uint8_t> {};

template<>
struct hash<kmm::BufferId>: hash<uint64_t> {};

template<>
struct hash<kmm::EventId>: hash<uint64_t> {};

}  // namespace std
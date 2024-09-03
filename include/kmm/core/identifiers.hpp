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

static constexpr size_t MAX_DEVICES = 4;
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
    constexpr MemoryId(DeviceId device) : MemoryId(device.get()) {}

    static constexpr MemoryId host() {
        return MemoryId {HOST_ID};
    }

    constexpr bool is_host() const {
        return m_value == HOST_ID;
    }

    constexpr bool is_device() const {
        return m_value != HOST_ID;
    }

    constexpr DeviceId as_device() const {
        KMM_ASSERT(is_device());
        return DeviceId(m_value);
    }

    constexpr bool operator==(const MemoryId& that) const {
        return m_value == that.m_value;
    }

    constexpr bool operator<(const MemoryId& that) const {
        // We assume the order Host, Device(0), Device(1), Device(2), ...
        if (is_host() || that.is_host()) {
            return that.is_device();
        } else {
            return m_value < that.m_value;
        }
    }

    KMM_IMPL_COMPARISON_OPS(MemoryId)

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

template<>
struct std::hash<kmm::NodeId>: std::hash<uint8_t> {};

template<>
struct std::hash<kmm::DeviceId>: std::hash<uint8_t> {};

template<>
struct std::hash<kmm::BufferId>: std::hash<uint64_t> {};

template<>
struct std::hash<kmm::EventId>: std::hash<uint64_t> {};

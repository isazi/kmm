#pragma once

#include <iosfwd>
#include <utility>

#include "fmt/ostream.h"

#include "kmm/core/macros.hpp"
#include "kmm/utils/checked_math.hpp"
#include "kmm/utils/panic.hpp"
#include "kmm/utils/small_vector.hpp"

#define KMM_IMPL_COMPARISON_OPS(T)                              \
    KMM_INLINE constexpr bool operator!=(const T& that) const { \
        return !(*this == that);                                \
    }                                                           \
    KMM_INLINE constexpr bool operator<=(const T& that) const { \
        return !(*this > that);                                 \
    }                                                           \
    KMM_INLINE constexpr bool operator>(const T& that) const {  \
        return that < *this;                                    \
    }                                                           \
    KMM_INLINE constexpr bool operator>=(const T& that) const { \
        return that <= *this;                                   \
    }

namespace kmm {

using index_t = int;

struct NodeId {
    explicit constexpr NodeId(uint8_t v) : m_value(v) {}
    explicit constexpr NodeId(size_t v) : m_value(checked_cast<uint8_t>(v)) {}

    KMM_INLINE constexpr uint8_t get() const {
        return m_value;
    }

    KMM_INLINE operator uint8_t() const {
        return get();
    }

    KMM_INLINE constexpr bool operator==(const NodeId& that) const {
        return m_value == that.m_value;
    }

    KMM_INLINE constexpr bool operator<(const NodeId& that) const {
        return m_value < that.m_value;
    }

    KMM_IMPL_COMPARISON_OPS(NodeId)

    friend std::ostream& operator<<(std::ostream&, const NodeId&);

  private:
    uint8_t m_value;
};

// Maximum of 4 devices per node
static constexpr size_t MAX_DEVICES = 4;

struct DeviceId {
    template<typename T>
    explicit constexpr DeviceId(T v) : m_value(static_cast<uint8_t>(v)) {
        if (!in_range(v, MAX_DEVICES)) {
            throw std::runtime_error("device index out of range");
        }
    }

    KMM_INLINE constexpr uint8_t get() const {
        if (m_value >= MAX_DEVICES) {
            __builtin_unreachable();
        }

        return m_value;
    }

    KMM_INLINE operator uint8_t() const {
        return get();
    }

    KMM_INLINE constexpr bool operator==(const DeviceId& that) const {
        return m_value == that.m_value;
    }

    KMM_INLINE constexpr bool operator<(const DeviceId& that) const {
        return m_value < that.m_value;
    }

    KMM_IMPL_COMPARISON_OPS(DeviceId)

    friend std::ostream& operator<<(std::ostream&, const DeviceId&);

  private:
    uint8_t m_value;
};

struct MemoryId {
  private:
    constexpr MemoryId(uint8_t v) : m_value(v) {}

  public:
    KMM_INLINE constexpr MemoryId(DeviceId device) : MemoryId(device.get()) {}

    KMM_INLINE static constexpr MemoryId host() {
        return MemoryId {HOST_ID};
    }

    KMM_INLINE constexpr bool is_host() const noexcept {
        return m_value == HOST_ID;
    }

    KMM_INLINE constexpr bool is_device() const noexcept {
        return m_value != HOST_ID;
    }

    KMM_INLINE constexpr DeviceId as_device() const noexcept {
        KMM_ASSERT(is_device());
        return DeviceId(m_value);
    }

    KMM_INLINE constexpr bool operator==(const MemoryId& that) const {
        return m_value == that.m_value;
    }

    KMM_INLINE constexpr bool operator<(const MemoryId& that) const {
        // We assume the order: Host, Device 0, Device 1, Device 2, ...
        if (is_host() || that.is_host()) {
            return that.is_device();
        } else {
            return m_value < that.m_value;
        }
    }

    KMM_IMPL_COMPARISON_OPS(MemoryId)

    friend std::ostream& operator<<(std::ostream&, const MemoryId&);

  private:
    static constexpr uint8_t HOST_ID = 0xff;
    uint8_t m_value = HOST_ID;
};

class ProcessorId {
  public:
    enum struct Type : uint8_t { Host, Cuda };

    KMM_INLINE constexpr ProcessorId(Type type, uint8_t id = 0) : m_type(type), m_value(id) {}

    KMM_INLINE constexpr ProcessorId(DeviceId device) : ProcessorId(Type::Cuda, device.get()) {}

    KMM_INLINE static constexpr ProcessorId host() {
        return ProcessorId {Type::Host};
    }

    KMM_INLINE constexpr bool is_host() const {
        return m_type == Type::Host;
    }

    KMM_INLINE constexpr bool is_device() const {
        return m_type == Type::Cuda;
    }

    KMM_INLINE constexpr DeviceId as_device() const {
        KMM_ASSERT(is_device());
        return DeviceId(m_value);
    }

    KMM_INLINE constexpr MemoryId as_memory() const {
        return is_host() ? MemoryId::host() : MemoryId(DeviceId(m_value));
    }

    KMM_INLINE constexpr bool operator==(const ProcessorId& that) const {
        return m_type == that.m_type && m_value == that.m_value;
    }

    KMM_INLINE constexpr bool operator<(const ProcessorId& that) const {
        if (m_type != that.m_type) {
            return m_type < that.m_type;
        } else {
            return m_value < that.m_value;
        }
    }

    KMM_IMPL_COMPARISON_OPS(ProcessorId)

    friend std::ostream& operator<<(std::ostream&, const ProcessorId&);

  private:
    Type m_type = Type::Host;
    uint8_t m_value = 0;
};

struct BufferId {
    KMM_INLINE explicit constexpr BufferId(uint64_t v = ~uint64_t(0)) : m_value(v) {}

    KMM_INLINE constexpr uint64_t get() const {
        return m_value;
    }

    KMM_INLINE operator uint64_t() const {
        return get();
    }

    KMM_INLINE constexpr bool operator==(const BufferId& that) const {
        return m_value == that.m_value;
    }

    KMM_INLINE constexpr bool operator<(const BufferId& that) const {
        return m_value < that.m_value;
    }

    KMM_IMPL_COMPARISON_OPS(BufferId)

    friend std::ostream& operator<<(std::ostream&, const BufferId&);

  private:
    uint64_t m_value;
};

struct EventId {
    KMM_INLINE explicit constexpr EventId(uint64_t v = 0) : m_value(v) {}

    KMM_INLINE constexpr uint64_t get() const {
        return m_value;
    }

    KMM_INLINE operator uint64_t() const {
        return get();
    }

    KMM_INLINE constexpr bool operator==(const EventId& that) const {
        return m_value == that.m_value;
    }

    KMM_INLINE constexpr bool operator<(const EventId& that) const {
        return m_value < that.m_value;
    }

    KMM_IMPL_COMPARISON_OPS(EventId)

    friend std::ostream& operator<<(std::ostream&, const EventId&);

  private:
    uint64_t m_value;
};

using EventList = small_vector<EventId, 2>;
std::ostream& operator<<(std::ostream&, const EventList&);

}  // namespace kmm

template<>
struct std::hash<kmm::NodeId>: std::hash<uint8_t> {};
template<>
struct std::hash<kmm::DeviceId>: std::hash<uint8_t> {};
template<>
struct std::hash<kmm::BufferId>: std::hash<uint64_t> {};
template<>
struct std::hash<kmm::EventId>: std::hash<uint64_t> {};

template<>
struct fmt::formatter<kmm::NodeId>: fmt::formatter<uint64_t> {};
template<>
struct fmt::formatter<kmm::DeviceId>: fmt::formatter<uint64_t> {};
template<>
struct fmt::formatter<kmm::BufferId>: fmt::formatter<uint64_t> {};
template<>
struct fmt::formatter<kmm::EventId>: fmt::formatter<uint64_t> {};
template<>
struct fmt::formatter<kmm::MemoryId>: fmt::ostream_formatter {};
template<>
struct fmt::formatter<kmm::ProcessorId>: fmt::ostream_formatter {};
template<>
struct fmt::formatter<kmm::EventList>: fmt::ostream_formatter {};

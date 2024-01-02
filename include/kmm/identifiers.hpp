#pragma once

#include <stddef.h>

#include "fmt/format.h"

#include "kmm/event.hpp"

// Macro for implementing comparison operators for class `T`
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

class MemoryId {
  public:
    explicit constexpr MemoryId(uint8_t value) : m_value(value) {}

    static constexpr MemoryId invalid() {
        return MemoryId(~uint8_t(0));
    }

    constexpr uint8_t get() const {
        return m_value;
    }

    operator uint8_t() const {
        return get();
    }

    constexpr bool operator==(const MemoryId& that) const {
        return m_value == that.m_value;
    }

    constexpr bool operator<(const MemoryId& that) const {
        return m_value < that.m_value;
    }

    KMM_IMPL_COMPARISON_OPS(MemoryId)

  private:
    uint8_t m_value;
};

class ExecutorId {
  public:
    explicit constexpr ExecutorId(uint8_t value) : m_value(value) {}

    constexpr uint8_t get() const {
        return m_value;
    }

    constexpr operator uint8_t() const {
        return get();
    }

    constexpr bool operator==(const ExecutorId& that) const {
        return m_value == that.m_value;
    }

    constexpr bool operator<(const ExecutorId& that) const {
        return m_value < that.m_value;
    }

    KMM_IMPL_COMPARISON_OPS(ExecutorId)

  private:
    uint8_t m_value;
};

class BlockId {
  public:
    explicit constexpr BlockId(EventId event, uint8_t index) : m_event(event), m_index(index) {}

    static constexpr BlockId invalid() {
        return BlockId(EventId::invalid(), 0);
    }

    constexpr uint8_t index() const {
        return m_index;
    }

    constexpr EventId event() const {
        return m_event;
    }

    constexpr size_t hash() const {
        return m_event.get() * 256 * m_index;
    }

    std::string to_string() const {
        return std::to_string(m_event.get()) + "@" + std::to_string(m_index);
    }

    constexpr bool operator==(const BlockId& that) const {
        return m_event == that.m_event && m_index == that.m_index;
    }

    constexpr bool operator!=(const BlockId& that) const {
        return !(*this == that);
    }

  private:
    EventId m_event;
    uint8_t m_index;
};

class BufferId {
  public:
    explicit constexpr BufferId(uint64_t index, size_t num_bytes) :
        m_index(index),
        m_nbytes(num_bytes) {}

    static constexpr BufferId invalid() {
        return BufferId(std::numeric_limits<uint64_t>::max(), 0);
    }

    constexpr uint64_t get() const {
        return m_index;
    }

    operator uint64_t() const {
        return m_index;
    }

    constexpr size_t num_bytes() const {
        return m_nbytes;
    }

    constexpr size_t hash() const {
        return m_index;
    }

    constexpr bool operator==(const BufferId& that) const {
        return m_index == that.m_index && m_nbytes == that.m_nbytes;
    }

    constexpr bool operator<(const BufferId& that) const {
        return m_index < that.m_index || (m_index == that.m_index && m_nbytes < that.m_nbytes);
    }

    KMM_IMPL_COMPARISON_OPS(BufferId)

  private:
    uint64_t m_index;
    size_t m_nbytes;
};
}  // namespace kmm

namespace fmt {
template<>
struct formatter<kmm::MemoryId>: formatter<uint8_t> {};

template<>
struct formatter<kmm::ExecutorId>: formatter<uint8_t> {};

template<>
struct formatter<kmm::BufferId>: formatter<uint64_t> {};

template<>
struct formatter<kmm::BlockId>: formatter<std::string> {
    template<typename FormatContext>
    auto format(const kmm::BlockId& bid, FormatContext& ctx) const {
        return fmt::format_to(ctx.out(), "{}@{}", bid.event(), bid.index());
    }
};
}  // namespace fmt

namespace std {
template<>
struct hash<kmm::MemoryId>: hash<uint8_t> {};

template<>
struct hash<kmm::ExecutorId>: hash<uint8_t> {};

template<>
struct hash<kmm::BufferId>: hash<uint64_t> {};

template<>
struct hash<kmm::BlockId> {
    size_t operator()(kmm::BlockId b) const {
        return b.hash();
    }
};
}  // namespace std
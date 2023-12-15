#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "fmt/format.h"

namespace kmm {

class RuntimeImpl;
class Runtime;
class Object;
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

    bool operator!=(const Identifier& that) const {
        return !(*this == that);
    }

    bool operator<=(const Identifier& that) const {
        return !(this > that);
    }

    bool operator>(const Identifier& that) const {
        return that < *this;
    }

    bool operator>=(const Identifier& that) const {
        return that <= *this;
    }

    Identifier operator++() {
        return Identifier(++m_value);
    }

    Identifier operator++(int) {
        return Identifier(m_value++);
    }

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
using BufferId = Identifier<uint64_t, struct BufferTag>;
using BlockId = Identifier<uint64_t, struct BlockTag>;

struct TaskInput {
    DeviceId memory_id;
    BlockId block_id;
};

struct TaskOutput {
    DeviceId memory_id;
    BlockId block_id;
    std::unique_ptr<BlockHeader> meta;
};

struct TaskRequirements {
    TaskRequirements(DeviceId id) : device_id(id) {}

    DeviceId device_id;
    std::vector<TaskInput> inputs = {};
    std::vector<TaskOutput> outputs = {};
};

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
        return &*events_.begin();
    }

    const EventId* end() const {
        return begin() + size();
    }

    EventId operator[](size_t index) const {
        return *(begin() + index);
    }

    size_t size() const {
        return events_.size();
    }

    void extend(const EventId* data, size_t len) {
        events_.insert(events_.end(), data, data + len);
    }

    void extend(const EventList& that) {
        extend(that.begin(), that.size());
    }

    void push_back(EventId event) {
        events_.push_back(event);
    }

    void remove_duplicates() {
        std::sort(events_.begin(), events_.end());
        auto last_unique = std::unique(std::begin(events_), std::end(events_));
        events_.erase(last_unique, std::end(events_));
    }

  private:
    std::vector<EventId> events_ = {};
};

}  // namespace kmm

namespace std {
template<typename T, typename Tag>
struct hash<kmm::Identifier<T, Tag>>: hash<T> {};
}  // namespace std
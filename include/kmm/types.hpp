#pragma once

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "fmt/format.h"
#include "kmm/identifiers.hpp"


namespace kmm {

class RuntimeImpl;
class Runtime;
class BlockHeader;

using index_t = int;

class EventList {
  public:
    EventList() = default;
    EventList(EventId id) {
        push_back(id);
    }

    EventList(std::initializer_list<EventId> list) {
        extend(list.begin(), list.size());
    }

    EventList(const std::vector<EventId>& list) {
        extend(list.data(), list.size());
    }

    EventId* begin() {
        return &*m_events.begin();
    }

    EventId* end() {
        return begin() + size();
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

  private:
    std::vector<EventId> m_events = {};
};

enum class PollResult { Pending, Ready };

class Waker: public std::enable_shared_from_this<Waker> {
  public:
    virtual ~Waker() = default;
    virtual void trigger_wakeup(bool allow_progress = false) const = 0;
};

}  // namespace kmm

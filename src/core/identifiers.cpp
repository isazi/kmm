#include <algorithm>
#include <vector>

#include "kmm/core/identifiers.hpp"

namespace kmm {

std::ostream& operator<<(std::ostream& f, const NodeId& v) {
    // Must cast to uin32_t, since uint8_t is formatted as a character (`char`)
    return f << uint32_t(v.get());
}

std::ostream& operator<<(std::ostream& f, const DeviceId& v) {
    // Must cast to uin32_t, since uint8_t is formatted as a character (`char`)
    return f << uint32_t(v.get());
}

std::ostream& operator<<(std::ostream& f, const MemoryId& v) {
    if (v.is_host()) {
        return f << "RAM";
    } else {
        return f << "GPU:" << v.as_device();
    }
}

std::ostream& operator<<(std::ostream& f, const ProcessorId& v) {
    if (v.is_host()) {
        return f << "CPU";
    } else {
        return f << "GPU:" << v.as_device();
    }
}

std::ostream& operator<<(std::ostream& f, const BufferId& v) {
    return f << v.get();
}

std::ostream& operator<<(std::ostream& f, const EventId& v) {
    return f << v.get();
}

std::ostream& operator<<(std::ostream& f, const EventList& v) {
    if (v.size() == 0) {
        return f << "[]";
    }

    if (v.size() == 1) {
        return f << "[" << v[0] << "]";
    }

    std::vector<EventId> events = {v.begin(), v.end()};
    std::sort(events.begin(), events.end());

    auto it = std::unique(events.begin(), events.end());
    events.erase(it, events.end());

    f << "[";
    bool is_first = true;

    for (const auto& e : events) {
        if (!is_first) {
            f << ", ";
        }

        is_first = false;
        f << e;
    }

    f << "]";
    return f;
}

}  // namespace kmm
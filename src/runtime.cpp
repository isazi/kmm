#include <future>

#include "kmm/executor.hpp"
#include "kmm/platforms/host.hpp"
#include "kmm/runtime.hpp"
#include "kmm/runtime_impl.hpp"

namespace kmm {

EventId Event::id() const {
    return m_id;
}

bool Event::wait_until(typename std::chrono::system_clock::time_point deadline) const {
    return Event::m_runtime->query_event(m_id, deadline);
}

bool Event::is_done() const {
    return wait_until(std::chrono::time_point<std::chrono::system_clock> {});
}

bool Event::wait_for(typename std::chrono::system_clock::duration duration) const {
    return wait_until(std::chrono::system_clock::now() + duration);
}

void Event::wait() const {
    wait_until(std::chrono::time_point<std::chrono::system_clock>::max());
}

class Buffer::Lifetime {
  public:
    Lifetime(BlockId id, std::shared_ptr<RuntimeImpl> rt) : m_id(id), m_runtime(std::move(rt)) {}

    void destroy() {
        auto id = std::exchange(m_id, BlockId::invalid());

        if (id != BlockId::invalid()) {
            m_runtime->delete_block(id);
        }
    }

    ~Lifetime() {
        destroy();
    }

    BlockId m_id = BlockId::invalid();
    std::shared_ptr<RuntimeImpl> m_runtime;
};

Buffer::Buffer(BlockId id, std::shared_ptr<RuntimeImpl> runtime) :
    m_lifetime(std::make_shared<Lifetime>(id, std::move(runtime))) {}

BlockId Buffer::id() const {
    return m_lifetime->m_id;
}

Runtime Buffer::runtime() const {
    return m_lifetime->m_runtime;
}

void Buffer::destroy() const {
    m_lifetime->destroy();
}

Runtime::Runtime(std::shared_ptr<RuntimeImpl> impl) : m_impl(std::move(impl)) {
    KMM_ASSERT(m_impl);
}

Event Runtime::submit_task(
    std::shared_ptr<Task> task,
    TaskRequirements reqs,
    EventList dependencies) const {
    auto event_id = m_impl->submit_task(std::move(task), std::move(reqs), std::move(dependencies));
    return {event_id, m_impl};
}

Event Runtime::barrier() const {
    auto event_id = m_impl->submit_barrier();
    return {event_id, m_impl};
}

Event Runtime::join(std::vector<Event> events) const {
    if (events.size() == 1) {
        return events[0];
    }

    EventList deps;
    for (auto& event : events) {
        deps.push_back(event.id());
    }

    auto event_id = m_impl->join_events(deps);
    return {event_id, m_impl};
}

Runtime build_runtime() {
    auto host_executor = std::make_shared<ParallelExecutor>();
    std::vector<std::shared_ptr<Executor>> executors = {host_executor};
    std::shared_ptr<Memory> memory = std::make_shared<HostMemory>(host_executor);

    return std::make_shared<RuntimeImpl>(executors, memory);
}

}  // namespace kmm
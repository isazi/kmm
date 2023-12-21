#include <future>

#include "kmm/executor.hpp"
#include "kmm/platforms/host.hpp"
#include "kmm/runtime.hpp"
#include "kmm/runtime_impl.hpp"

namespace kmm {

Runtime::Runtime(std::shared_ptr<RuntimeImpl> impl) : m_impl(std::move(impl)) {
    KMM_ASSERT(m_impl);
}

EventId Runtime::submit_task(
    std::shared_ptr<Task> task,
    TaskRequirements reqs,
    EventList dependencies) const {
    return m_impl->submit_task(std::move(task), std::move(reqs), std::move(dependencies));
}

EventId Runtime::submit_barrier() const {
    return m_impl->submit_barrier();
}

EventId Runtime::join(EventList events) const {
    if (events.size() == 1) {
        return events[0];
    }

    return m_impl->join_events(std::move(events));
}

bool Runtime::wait_until(EventId id, typename std::chrono::system_clock::time_point deadline)
    const {
    return m_impl->query_event(id, deadline);
}

bool Runtime::is_done(EventId id) const {
    return wait_until(id, std::chrono::time_point<std::chrono::system_clock> {});
}

bool Runtime::wait_for(EventId id, typename std::chrono::system_clock::duration duration) const {
    return wait_until(id, std::chrono::system_clock::now() + duration);
}

void Runtime::wait(EventId id) const {
    wait_until(id, std::chrono::time_point<std::chrono::system_clock>::max());
}

void Runtime::synchronize() const {
    m_impl->query_event(m_impl->submit_barrier());
}

Runtime build_runtime() {
    auto host_executor = std::make_shared<ParallelExecutor>();
    std::vector<std::shared_ptr<Executor>> executors = {host_executor};
    std::shared_ptr<Memory> memory = std::make_shared<HostMemory>(host_executor);

    return std::make_shared<RuntimeImpl>(executors, memory);
}

Buffer::Buffer(std::shared_ptr<RuntimeImpl> runtime, BlockId id) :
    m_id(id),
    m_runtime(std::move(runtime)) {
    KMM_ASSERT(m_runtime != nullptr);
}

Buffer::~Buffer() {
    destroy();
}

EventId Buffer::prefetch(MemoryId memory_id, EventList dependencies) const {
    return m_runtime->submit_block_prefetch(m_id, memory_id, std::move(dependencies));
}

EventId Buffer::submit_barrier() const {
    return m_runtime->submit_block_barrier(m_id);
}

void Buffer::synchronize() const {
    m_runtime->query_event(submit_barrier(), std::chrono::system_clock::time_point::max());
}

void Buffer::destroy() {
    if (m_id != BlockId::invalid()) {
        m_runtime->delete_block(m_id);
        m_id = BlockId::invalid();
    }
}

}  // namespace kmm
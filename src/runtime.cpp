#include <future>

#include "kmm/executor.hpp"
#include "kmm/platforms/host.hpp"
#include "kmm/runtime.hpp"
#include "kmm/runtime_impl.hpp"

namespace kmm {

OperationId Event::id() const {
    return m_id;
}

void Event::wait() const {
    std::promise<void> promise;
    auto future = promise.get_future();
    m_runtime->submit_promise(m_id, std::move(promise));
    future.get();
}

class Buffer::Lifetime {
  public:
    Lifetime(BufferId id, std::shared_ptr<RuntimeImpl> rt) : m_id(id), m_runtime(std::move(rt)) {}

    void destroy() {
        auto id = std::exchange(m_id, BufferId::invalid());

        if (id != BufferId::invalid()) {
            m_runtime->delete_buffer(id);
        }
    }

    ~Lifetime() {
        destroy();
    }

    BufferId m_id = BufferId::invalid();
    std::shared_ptr<RuntimeImpl> m_runtime;
};

Buffer::Buffer(BufferId id, std::shared_ptr<RuntimeImpl> rt) :
    m_lifetime(std::make_shared<Lifetime>(id, std::move(rt))) {}

BufferId Buffer::id() const {
    return m_lifetime->m_id;
}

Runtime Buffer::runtime() const {
    return m_lifetime->m_runtime;
}

Event Buffer::barrier() const {
    auto event_id = m_lifetime->m_runtime->submit_buffer_barrier(id());
    return {event_id, m_lifetime->m_runtime};
}

void Buffer::destroy() const {
    m_lifetime->destroy();
}

Runtime::Runtime(std::shared_ptr<RuntimeImpl> impl) : m_impl(std::move(impl)) {
    KMM_ASSERT(m_impl);
}

Buffer Runtime::allocate_buffer(
    size_t num_elements,
    size_t element_size,
    size_t element_align,
    DeviceId home) const {
    size_t num_bytes = element_size * num_elements;

    auto id = m_impl->create_buffer(BufferLayout {
        .num_bytes = num_bytes,
        .alignment = element_align,
        .home = home,
        .name = ""});

    return {id, m_impl};
}

void Runtime::submit_task(
    std::shared_ptr<Task> task,
    TaskRequirements reqs,
    std::vector<OperationId> dependencies) const {
    m_impl->submit_task(std::move(task), std::move(reqs), std::move(dependencies));
}

Event Runtime::barrier() const {
    auto event_id = m_impl->submit_barrier();
    return {event_id, m_impl};
}

Event Runtime::barrier_buffer(BufferId buffer_id) const {
    auto event_id = m_impl->submit_buffer_barrier(buffer_id);
    return {event_id, m_impl};
}

Runtime build_runtime() {
    auto host_executor = std::make_shared<ParallelExecutor>();
    std::vector<std::shared_ptr<Executor>> executors = {host_executor};
    std::shared_ptr<Memory> memory = std::make_shared<HostMemory>(host_executor);

    return std::make_shared<RuntimeImpl>(executors, memory);
}

}  // namespace kmm
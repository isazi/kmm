#include <future>

#include "kmm/cuda/executor.hpp"
#include "kmm/cuda/memory.hpp"
#include "kmm/cuda/types.hpp"
#include "kmm/executor.hpp"
#include "kmm/host/executor.hpp"
#include "kmm/host/memory.hpp"
#include "kmm/runtime.hpp"
#include "kmm/runtime_impl.hpp"

namespace kmm {

Runtime::Runtime(std::shared_ptr<RuntimeImpl> impl) : m_impl(std::move(impl)) {
    KMM_ASSERT(m_impl);
}

EventId Runtime::submit_task(std::shared_ptr<Task> task, TaskRequirements reqs) const {
    return m_impl->submit_task(std::move(task), std::move(reqs));
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
    auto host_executor = std::make_shared<ParallelExecutorHandle>();

    std::vector<std::shared_ptr<ExecutorHandle>> executors = {host_executor};
    std::unique_ptr<Memory> memory;

    auto devices = get_cuda_devices();
    if (!devices.empty()) {
        auto contexts = std::vector<CudaContextHandle> {};
        uint8_t memory_id = 1;

        for (auto device : devices) {
            auto context = CudaContextHandle::from_new_context(device);
            contexts.push_back(context);

            executors.push_back(std::make_shared<CudaExecutorHandle>(context, MemoryId(memory_id)));
            memory_id++;
        }

        memory = std::make_unique<CudaMemory>(host_executor, contexts);
    } else {
        memory = std::make_unique<HostMemory>(host_executor);
    }

    return std::make_shared<RuntimeImpl>(std::move(executors), std::move(memory));
}

}  // namespace kmm
#include <utility>

#include "kmm/runtime_impl.hpp"
#include "kmm/scheduler.hpp"

namespace kmm {

RuntimeImpl::RuntimeImpl(
    std::vector<std::shared_ptr<Executor>> executors,
    std::shared_ptr<Memory> memory) :
    m_scheduler(std::make_shared<Scheduler>(
        executors,
        std::make_shared<MemoryManager>(memory),
        std::make_shared<BlockManager>())),
    m_thread(m_scheduler) {}

RuntimeImpl::~RuntimeImpl() {
    m_scheduler->shutdown();
    m_thread.join();
}

EventId RuntimeImpl::submit_task(std::shared_ptr<Task> task, TaskRequirements reqs, EventList deps)
    const {
    std::lock_guard guard {m_mutex};

    for (auto& input : reqs.inputs) {
        auto accesses = m_block_accesses.at(input.block_id);
        deps.extend(accesses);
    }

    auto id = EventId(m_next_event++);
    m_scheduler->submit_command(
        id,
        CommandExecute {
            .device_id = reqs.device_id,
            .task = std::move(task),
            .inputs = std::move(reqs.inputs),
            .outputs = std::move(reqs.outputs),
        },
        std::move(deps));

    return id;
}

EventId RuntimeImpl::delete_block(BlockId block_id, EventList deps) const {
    std::lock_guard guard {m_mutex};

    auto& accesses = m_block_accesses[block_id];
    deps.extend(accesses);

    auto id = EventId(m_next_event++);
    m_scheduler->submit_command(id, CommandBlockDelete {block_id}, std::move(deps));

    m_block_accesses.erase(block_id);
    return id;
}

EventId RuntimeImpl::join_events(EventList deps) const {
    std::lock_guard guard {m_mutex};

    auto id = EventId(m_next_event++);
    m_scheduler->submit_command(id, CommandNoop {}, std::move(deps));
    return id;
}

EventId RuntimeImpl::submit_barrier() const {
    std::lock_guard guard {m_mutex};

    auto deps = EventList {};
    auto id = EventId(m_next_event++);

    for (auto& entry : m_block_accesses) {
        auto& accesses = entry.second;
        deps.extend(accesses);
        accesses = {id};
    }

    return id;
}

bool RuntimeImpl::query_event(
    EventId id,
    std::chrono::time_point<std::chrono::system_clock> deadline) const {
    // No need to get the lock, the scheduler has its own internal lock
    //std::lock_guard guard {m_mutex};

    return m_scheduler->query_event(id, deadline);
}

}  // namespace kmm
#include <functional>
#include <stdexcept>
#include <utility>

#include "fmt/ranges.h"
#include "spdlog/spdlog.h"

#include "kmm/jobs.hpp"
#include "kmm/memory_manager.hpp"
#include "kmm/utils.hpp"
#include "kmm/worker.hpp"

namespace kmm {

void WorkerJob::trigger_wakeup() const {
    if (auto owner = worker.lock()) {
        owner->wakeup(const_cast<WorkerJob*>(this)->shared_from_this());
    }
}

void WorkerJob::trigger_wakeup_and_poll() const {
    if (auto owner = worker.lock()) {
        owner->wakeup(const_cast<WorkerJob*>(this)->shared_from_this(), true);
    }
}

bool JobQueue::is_empty() const {
    return m_head == nullptr;
}

bool JobQueue::push(std::shared_ptr<WorkerJob> op) {
    if (op->in_queue.test_and_set()) {
        return false;
    }

    op->next_item = nullptr;

    if (m_head == nullptr) {
        m_tail = op.get();
        m_head = std::move(op);

    } else {
        auto* old_tail = std::exchange(m_tail, op.get());
        old_tail->next_item = std::move(op);
    }

    return true;
}

std::optional<std::shared_ptr<WorkerJob>> JobQueue::pop() {
    if (m_head == nullptr) {
        return std::nullopt;
    }

    auto op = std::move(m_head);
    m_head = std::move(op->next_item);

    op->in_queue.clear();
    return std::optional {std::move(op)};
}

bool SharedJobQueue::push(std::shared_ptr<WorkerJob> op) const {
    std::lock_guard guard {m_lock};
    bool needs_notify = m_queue.is_empty();

    if (m_queue.push(std::move(op))) {
        if (needs_notify) {
            m_cond.notify_one();
        }

        return true;
    }

    return false;
}

JobQueue SharedJobQueue::pop_all(
    std::chrono::time_point<std::chrono::system_clock> deadline) const {
    std::unique_lock guard {m_lock};
    if (deadline != std::chrono::time_point<std::chrono::system_clock> {}) {
        while (m_queue.is_empty()) {
            if (m_cond.wait_until(guard, deadline) == std::cv_status::timeout) {
                break;
            }
        }
    }

    return std::move(m_queue);
}

Worker::Worker(
    std::vector<std::shared_ptr<Executor>> executors,
    std::shared_ptr<MemoryManager> memory_manager) :
    m_state {executors, memory_manager, BlockManager()} {}

void Worker::make_progress(std::chrono::time_point<std::chrono::system_clock> deadline) {
    make_progress_impl(std::unique_lock {m_lock}, deadline);
}

void Worker::wakeup(std::shared_ptr<WorkerJob> op, bool allow_progress) {
    if (allow_progress) {
        if (auto guard = std::unique_lock {m_lock, std::try_to_lock}) {
            m_poll_queue.push(std::move(op));
            make_progress_impl(std::move(guard));
            return;
        }
    }

    m_shared_poll_queue.push(std::move(op));
}

void Worker::submit_command(EventId id, Command command, EventList dependencies) {
    std::unique_lock guard {m_lock};

    if (m_shutdown) {
        throw std::runtime_error("cannot submit new commands after shutdown");
    }

    spdlog::debug("submit command id={} dependencies={} command={}", id, dependencies, command);
    std::shared_ptr<WorkerJob> op;

    if (auto* cmd = std::get_if<CommandExecute>(&command)) {
        op = std::make_shared<ExecuteJob>(id, std::move(*cmd));
    } else if (auto* cmd = std::get_if<CommandBlockDelete>(&command)) {
        op = std::make_shared<DeleteJob>(id, cmd->id);
    } else if (std::get_if<CommandNoop>(&command)) {
        op = std::make_shared<EmptyJob>(id);
    } else if (const auto* cmd = std::get_if<CommandPrefetch>(&command)) {
        op = std::make_shared<PrefetchJob>(id, cmd->device_id, cmd->block_id);
    } else {
        KMM_PANIC("invalid command");
    }

    KMM_ASSERT(op->status == WorkerJob::Status::Created);
    op->status = WorkerJob::Status::Pending;
    op->unsatisfied_predecessors = dependencies.size() + 1;

    dependencies.remove_duplicates();
    size_t satisfied = 1;

    for (auto dep_id : dependencies) {
        auto it = m_jobs.find(dep_id);

        if (it != m_jobs.end()) {
            auto predecessor = it->second;
            predecessor->successors.push_back(op);
        } else {
            satisfied++;
        }
    }

    satisfy_job_dependencies(std::move(op), satisfied);
    make_progress_impl(std::move(guard));
}

void Worker::satisfy_job_dependencies(std::shared_ptr<WorkerJob> op, size_t satisfied) {
    if (op->unsatisfied_predecessors > satisfied) {
        op->unsatisfied_predecessors -= satisfied;
        return;
    }

    // Job is now ready!
    op->unsatisfied_predecessors = 0;
    op->status = WorkerJob::Status::Ready;
    m_ready_queue.push(std::move(op));
}

bool Worker::query_event(EventId id, std::chrono::time_point<std::chrono::system_clock> deadline) {
    std::unique_lock<std::mutex> guard {m_lock};

    while (true) {
        if (m_jobs.find(id) == m_jobs.end()) {
            return true;
        }

        if (deadline == decltype(deadline)()) {
            return false;
        }

        if (m_job_completion.wait_until(guard, deadline) == std::cv_status::timeout) {
            return false;
        }
    }
}

void Worker::shutdown() {
    std::unique_lock<std::mutex> guard {m_lock};
    m_shutdown = true;
}

bool Worker::is_shutdown() {
    std::unique_lock<std::mutex> guard {m_lock};
    return m_shutdown && m_jobs.empty();
}

void Worker::make_progress_impl(
    std::unique_lock<std::mutex> guard,
    std::chrono::time_point<std::chrono::system_clock> deadline) {
    if (m_poll_queue.is_empty()) {
        m_poll_queue = m_shared_poll_queue.pop_all();

        if (m_poll_queue.is_empty() && deadline != decltype(deadline)()) {
            guard.unlock();
            m_poll_queue = m_shared_poll_queue.pop_all(deadline);
            guard.lock();
        }
    }

    while (true) {
        if (!m_poll_queue.is_empty()) {
            while (auto op = m_poll_queue.pop()) {
                poll_job(*op);
            }

            m_poll_queue = m_shared_poll_queue.pop_all();
            continue;
        }

        if (auto op = m_ready_queue.pop()) {
            start_job(*op);
            continue;
        }

        break;
    }
}

void Worker::start_job(std::shared_ptr<WorkerJob>& op) {
    KMM_ASSERT(op->status == WorkerJob::Status::Ready);
    op->status = WorkerJob::Status::Running;
    op->worker = shared_from_this();
    op->start(m_state);

    poll_job(op);
}

void Worker::poll_job(const std::shared_ptr<WorkerJob>& op) {
    // Status has to be `Running`
    if (op->status == WorkerJob::Status::Running) {
        if (op->poll(m_state) == PollResult::Ready) {
            stop_job(op);
        }
    }
}

void Worker::stop_job(const std::shared_ptr<WorkerJob>& op) {
    spdlog::debug("removing job id={}", op->id());
    op->status = WorkerJob::Status::Done;
    op->worker.reset();
    op->stop(m_state);

    auto successors = std::move(op->successors);
    for (auto& successor : successors) {
        satisfy_job_dependencies(std::move(successor));
    }

    m_jobs.erase(op->id());
    m_job_completion.notify_all();
}

}  // namespace kmm

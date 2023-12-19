#include <functional>
#include <stdexcept>
#include <utility>

#include "fmt/ranges.h"
#include "spdlog/spdlog.h"

#include "kmm/utils.hpp"
#include "kmm/worker/jobs.hpp"
#include "kmm/worker/memory_manager.hpp"
#include "kmm/worker/worker.hpp"

namespace kmm {

void Job::trigger_wakeup(bool allow_progress) const {
    if (auto owner = worker.lock()) {
        auto self = std::shared_ptr<Job>(shared_from_this(), const_cast<Job*>(this));
        owner->wakeup(self, allow_progress);
    }
}

bool JobQueue::is_empty() const {
    return m_head == nullptr;
}

bool JobQueue::push(std::shared_ptr<Job> op) {
    if (op->in_queue.test_and_set()) {
        return false;
    }

    op->next_queue_item = nullptr;

    if (m_head == nullptr) {
        m_tail = op.get();
        m_head = std::move(op);
    } else {
        auto* old_tail = std::exchange(m_tail, op.get());
        old_tail->next_queue_item = std::move(op);
    }

    return true;
}

std::optional<std::shared_ptr<Job>> JobQueue::pop() {
    if (m_head == nullptr) {
        return std::nullopt;
    }

    auto op = std::move(m_head);
    m_head = std::move(op->next_queue_item);

    op->in_queue.clear();
    return std::optional {std::move(op)};
}

bool SharedJobQueue::push(std::shared_ptr<Job> op) const {
    std::lock_guard guard {m_lock};
    auto id = op->id();
    bool needs_notify = m_queue.is_empty();

    if (m_queue.push(std::move(op))) {
        spdlog::debug("pushed id={}, needs_notify={}", id, needs_notify);
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

            spdlog::debug(
                "woke up! is_empty={} deadline={}",
                m_queue.is_empty(),
                deadline.time_since_epoch().count());
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

void Worker::wakeup(std::shared_ptr<Job> job, bool allow_progress) {
    auto id = job->id();

    if (allow_progress) {
        if (auto guard = std::unique_lock {m_lock, std::try_to_lock}) {
            bool result = m_local_poll_queue.push(std::move(job));
            spdlog::debug("add local poll id={} result={}", id, result);
            make_progress_impl(std::move(guard));
            return;
        }
    }

    auto result = m_shared_poll_queue.push(std::move(job));
    spdlog::debug("add shared poll id={} result={}", id, result);
}

void Worker::submit_command(EventId id, Command command, EventList dependencies) {
    std::unique_lock guard {m_lock};

    if (m_shutdown) {
        throw std::runtime_error("cannot submit new commands after shutdown");
    }

    spdlog::debug("submit command id={} dependencies={} command={}", id, dependencies, command);
    m_scheduler.insert(id, std::move(command), std::move(dependencies));
    make_progress_impl(std::move(guard));
}

bool Worker::query_event(EventId id, std::chrono::time_point<std::chrono::system_clock> deadline) {
    std::unique_lock<std::mutex> guard {m_lock};

    while (true) {
        spdlog::info("finding {}", id);
        if (m_scheduler.is_completed(id)) {
            spdlog::info("found {}!", id);
            return true;
        }

        if (m_shutdown || deadline == decltype(deadline)()) {
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
    return m_shutdown && m_scheduler.is_all_completed();
}

void Worker::make_progress_impl(
    std::unique_lock<std::mutex> guard,
    std::chrono::time_point<std::chrono::system_clock> deadline) {
    if (m_local_poll_queue.is_empty()) {
        auto result = m_shared_poll_queue.pop_all();

        if (result.is_empty() && deadline != decltype(deadline)()) {
            guard.unlock();
            result = m_shared_poll_queue.pop_all(deadline);
            guard.lock();
        }

        m_local_poll_queue = std::move(result);
    }

    while (true) {
        if (!m_local_poll_queue.is_empty()) {
            while (auto op = m_local_poll_queue.pop()) {
                poll_job(**op);
            }

            m_local_poll_queue = m_shared_poll_queue.pop_all();
            continue;
        }

        if (auto op = m_scheduler.pop_ready()) {
            start_job(std::move(*op));
            continue;
        }

        break;
    }
}

void Worker::start_job(std::shared_ptr<Scheduler::Node> node) {
    spdlog::debug("start job id={}", node->id());

    auto job = build_job_for_command(node->id(), node->take_command());

    KMM_ASSERT(job->status == Job::Status::Created);
    job->status = Job::Status::Running;
    job->worker = weak_from_this();
    job->completion = std::move(node);
    job->start(m_state);
    poll_job(*job);
}

void Worker::poll_job(Job& job) {
    if (job.status == Job::Status::Running) {
        spdlog::debug("poll job id={}", job.id());

        if (job.poll(m_state) == PollResult::Ready) {
            stop_job(job);
        }
    }
}

void Worker::stop_job(Job& job) {
    spdlog::debug("stop job id={}", job.id());
    job.status = Job::Status::Done;
    job.stop(m_state);

    m_scheduler.complete(std::move(job.completion));
    m_job_completion.notify_all();
}

}  // namespace kmm

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

    std::lock_guard guard {m_shared_poll_queue.m_lock};
    bool needs_notify = m_shared_poll_queue.m_queue.is_empty();

    if (m_shared_poll_queue.m_queue.push(std::move(job))) {
        if (needs_notify) {
            m_shared_poll_queue.m_cond.notify_one();
        }
    }
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
    bool should_wait = deadline != decltype(deadline)();

    while (true) {
        // First, drain the local poll queue
        if (auto op = m_local_poll_queue.pop()) {
            poll_job(**op);
            should_wait = false;
            continue;
        }

        m_local_poll_queue.push_all(m_shared_poll_queue.pop_all());
        if (!m_local_poll_queue.is_empty()) {
            should_wait = false;
            continue;
        }

        // Next, check if the scheduler has a job
        if (auto op = m_scheduler.pop_ready()) {
            start_job(std::move(*op));
            should_wait = false;
            continue;
        }

        if (should_wait) {
            guard.unlock();
            auto result = m_shared_poll_queue.pop_all(deadline);
            guard.lock();

            m_local_poll_queue.push_all(std::move(result));
            should_wait = false;
            continue;
        }

        break;
    }
}

void Worker::start_job(std::shared_ptr<Scheduler::Node> node) {
    auto command = node->take_command();

    // For empty commands, bypass the regular job procedure and immediately complete it.
    if (std::holds_alternative<EmptyCommand>(command)) {
        m_scheduler.complete(std::move(node));
        m_job_completion.notify_all();
        return;
    }

    spdlog::debug("start job id={}", node->id());
    auto job = build_job_for_command(node->id(), std::move(command));

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

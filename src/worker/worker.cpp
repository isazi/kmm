#include <functional>
#include <stdexcept>
#include <utility>

#include "fmt/ranges.h"
#include "spdlog/spdlog.h"

#include "kmm/panic.hpp"
#include "kmm/worker/jobs.hpp"
#include "kmm/worker/memory_manager.hpp"
#include "kmm/worker/worker.hpp"

namespace kmm {

Worker::Worker(std::vector<std::shared_ptr<Executor>> executors, std::unique_ptr<Memory> memory) :
    m_shared_poll_queue {std::make_shared<SharedJobQueue>()},
    m_state {executors, std::make_shared<MemoryManager>(std::move(memory)), BlockManager()} {}

void Worker::make_progress(std::chrono::time_point<std::chrono::system_clock> deadline) {
    while (true) {
        {
            std::unique_lock guard {m_lock};
            if (make_progress_impl()) {
                return;
            }
        }

        if (!m_shared_poll_queue->wait_until(deadline)) {
            return;
        }
    }
}

void Worker::wakeup(std::shared_ptr<Job> job, bool allow_progress) {
    auto id = job->id();

    if (allow_progress) {
        if (auto guard = std::unique_lock {m_lock, std::try_to_lock}) {
            bool result = m_local_poll_queue.push(std::move(job));
            spdlog::debug("add local poll id={} result={}", id, result);
            make_progress_impl();
            return;
        }
    }

    m_shared_poll_queue->push_job(std::move(job));
}

void Worker::submit_command(EventId id, Command command, EventList dependencies) {
    std::unique_lock guard {m_lock};

    if (m_shutdown) {
        throw std::runtime_error("cannot submit new commands after shutdown");
    }

    spdlog::debug("submit command id={} dependencies={} command={}", id, dependencies, command);
    m_scheduler.insert(id, std::move(command), std::move(dependencies));
    make_progress_impl();
}

bool Worker::query_event(EventId id, std::chrono::time_point<std::chrono::system_clock> deadline) {
    std::unique_lock<std::mutex> guard {m_lock};

    while (true) {
        if (m_scheduler.is_completed(id)) {
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
    m_job_completion.notify_all();
    m_shutdown = true;
}

bool Worker::is_shutdown() {
    std::unique_lock<std::mutex> guard {m_lock};
    return m_shutdown && m_scheduler.is_all_completed();
}

bool Worker::make_progress_impl() {
    bool progressed = false;

    while (true) {
        // First, drain the local poll queue
        if (auto op = m_local_poll_queue.pop()) {
            poll_job(**op);
            progressed = true;
            continue;
        }

        // Next, move from the shared poll queue to the local poll queue
        m_local_poll_queue.push_all(m_shared_poll_queue->pop_all_jobs());
        if (!m_local_poll_queue.is_empty()) {
            progressed = true;
            continue;
        }

        // Next, check if the scheduler has a job
        if (auto op = m_scheduler.pop_ready()) {
            start_job(std::move(*op));
            progressed = true;
            continue;
        }

        break;
    }

    return progressed;
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

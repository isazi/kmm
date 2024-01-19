#include <functional>
#include <stdexcept>
#include <utility>

#include "fmt/ranges.h"
#include "spdlog/spdlog.h"

#include "kmm/panic.hpp"
#include "kmm/utils/scope_guard.hpp"
#include "kmm/worker/jobs.hpp"
#include "kmm/worker/memory_manager.hpp"
#include "kmm/worker/worker.hpp"

namespace kmm {

Worker::Worker(std::vector<std::shared_ptr<DeviceHandle>> devices, std::unique_ptr<Memory> memory) :
    m_shared_poll_queue {std::make_shared<SharedJobQueue>()},
    m_state {devices, std::make_shared<MemoryManager>(std::move(memory)), BlockManager()} {}

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

void Worker::submit_barrier(EventId id) {
    submit_command(id, EmptyCommand {}, m_scheduler.active_tasks());
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
    auto&& command = node->take_command();

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

void Worker::create_block(
    BlockId block_id,
    MemoryId memory_id,
    std::unique_ptr<BlockHeader> header,
    const void* src_data,
    size_t num_bytes) {
    std::unique_lock guard {m_lock};

    auto layout = header->layout();
    auto waker = std::make_shared<ThreadWaker>();
    auto buffer_id = std::optional<BufferId> {};

    if (layout.num_bytes > 0) {
        buffer_id = m_state.memory_manager->create_buffer(layout);

        try {
            auto transaction = m_state.memory_manager->create_transaction(waker);
            auto request = m_state.memory_manager->create_request(  //
                *buffer_id,
                memory_id,
                AccessMode::Write,
                transaction);

            ScopeGuard delete_on_exit = [&] { m_state.memory_manager->delete_request(request); };

            guard.unlock();

            while (m_state.memory_manager->poll_request(request) == PollResult::Pending) {
                waker->wait();
            }

            auto* alloc = m_state.memory_manager->view_buffer(request);
            alloc->copy_from_host_sync(src_data, 0, num_bytes);
        } catch (...) {
            m_state.memory_manager->delete_buffer(*buffer_id);
            throw;
        }
    }

    m_state.block_manager.insert_block(block_id, std::move(header), memory_id, buffer_id);
}

std::shared_ptr<BlockHeader> Worker::read_block_header(BlockId block_id) {
    std::lock_guard guard {m_lock};
    const auto& meta = m_state.block_manager.get_block(block_id);
    return meta.header;
}

std::shared_ptr<BlockHeader> Worker::read_block(
    BlockId block_id,
    std::optional<MemoryId> preferred_memory_id,
    void* dst_data,
    size_t num_bytes) {
    std::unique_lock guard {m_lock};
    const auto& meta = m_state.block_manager.get_block(block_id);
    auto header = meta.header;
    auto buffer_id = meta.buffer_id;
    size_t buffer_size = buffer_id.has_value() ? buffer_id->num_bytes() : 0;

    if (buffer_size != num_bytes) {
        throw std::invalid_argument("invalid buffer size");
    }

    if (buffer_size == 0) {
        return header;
    }

    auto memory_id = preferred_memory_id.has_value() ? *preferred_memory_id : meta.home_memory;

    auto waker = std::make_shared<ThreadWaker>();
    auto transaction = m_state.memory_manager->create_transaction(waker);
    auto request = m_state.memory_manager
                       ->create_request(*buffer_id, memory_id, AccessMode::Read, transaction);

    ScopeGuard delete_on_exit = [&] { m_state.memory_manager->delete_request(request); };

    guard.unlock();

    while (m_state.memory_manager->poll_request(request) == PollResult::Pending) {
        waker->wait();
    }

    const auto* alloc = m_state.memory_manager->view_buffer(request);
    alloc->copy_to_host_sync(0, dst_data, num_bytes);

    return header;
}

}  // namespace kmm

#include <functional>
#include <stdexcept>
#include <utility>

#include "fmt/ranges.h"
#include "spdlog/spdlog.h"

#include "kmm/memory_manager.hpp"
#include "kmm/utils.hpp"
#include "kmm/worker.hpp"

namespace kmm {

void WorkerJob::start(Worker& worker) {
    m_worker = worker.weak_from_this();
}

void WorkerJob::stop(Worker& worker) {
    m_worker.reset();
}

void WorkerJob::trigger_wakeup() const {
    if (auto owner = m_worker.lock()) {
        owner->wakeup(const_cast<WorkerJob*>(this)->shared_from_this());
    }
}

void WorkerJob::trigger_wakeup_and_poll() const {
    if (auto owner = m_worker.lock()) {
        owner->wakeup(const_cast<WorkerJob*>(this)->shared_from_this(), true);
    }
}

PollResult ExecuteJob::poll(Worker& worker) {
    if (status == Status::Created) {
        auto requests = std::vector<MemoryRequest>();

        for (const auto& arg : inputs) {
            auto buffer_id_opt = worker.m_block_manager.get_block_buffer(arg.block_id);

            if (buffer_id_opt) {
                requests.emplace_back(worker.m_memory_manager->create_request(  //
                    *buffer_id_opt,
                    arg.memory_id,
                    false,
                    shared_from_this()));
            } else {
                requests.emplace_back(nullptr);
            }
        }

        for (const auto& arg : outputs) {
            auto layout = arg.meta->layout();

            if (layout.num_bytes > 0) {
                auto buffer_id = worker.m_memory_manager->create_buffer(arg.meta->layout());

                output_buffers.emplace_back(buffer_id);
                requests.emplace_back(worker.m_memory_manager->create_request(  //
                    buffer_id,
                    arg.memory_id,
                    true,
                    shared_from_this()));
            } else {
                output_buffers.emplace_back(std::nullopt);
                requests.emplace_back(nullptr);
            }
        }

        status = Status::Staging;
        memory_requests = std::move(requests);
    }

    if (status == Status::Staging) {
        if (worker.m_memory_manager->poll_requests(memory_requests) != PollResult::Ready) {
            return PollResult::Pending;
        }

        try {
            auto context = TaskContext {};
            size_t index = 0;

            for (const auto& input : inputs) {
                const auto& req = memory_requests[index++];
                auto header = worker.m_block_manager.get_block_header(input.block_id);
                const auto* allocation = req ? worker.m_memory_manager->view_buffer(req) : nullptr;

                context.inputs.push_back(InputBlock {
                    .block_id = input.block_id,
                    .header = header,
                    .allocation = allocation,
                });
            }

            for (const auto& output : outputs) {
                const auto& req = memory_requests[index++];
                const auto* allocation = req ? worker.m_memory_manager->view_buffer(req) : nullptr;

                context.outputs.push_back(OutputBuffer {
                    .block_id = output.block_id,
                    .header = output.meta.get(),
                    .allocation = allocation,
                });
            }

            // Mmmmm, do we need the cast here?
            auto completion = std::dynamic_pointer_cast<ExecuteJob>(shared_from_this());

            worker.m_executors.at(device_id)->submit(
                task,
                std::move(context),
                TaskCompletion(std::move(completion)));
        } catch (const std::exception& e) {
            result = TaskError(e);
        }

        status = Status::Running;
    }

    if (status == Status::Running) {
        if (!result.has_value()) {
            return PollResult::Pending;
        }

        for (const auto& request : memory_requests) {
            if (request) {
                worker.m_memory_manager->delete_request(request);
            }
        }

        memory_requests.clear();

        size_t num_outputs = outputs.size();
        const auto* error = std::get_if<TaskError>(&*result);

        for (size_t i = 0; i < num_outputs; i++) {
            auto& output = outputs[i];
            auto block_id = output.block_id;
            auto buffer_id = output_buffers[i];

            if (error == nullptr) {
                worker.m_block_manager.insert_block(block_id, std::move(output.meta), buffer_id);
            } else {
                worker.m_block_manager.poison_block(block_id, *error);

                if (buffer_id) {
                    worker.m_memory_manager->delete_buffer(*buffer_id);
                }
            }
        }

        status = Status::Done;
    }

    return PollResult::Ready;
}

void ExecuteJob::complete_task(TaskResult result) {
    // Maybe we need some lock?
    this->result = std::move(result);

    trigger_wakeup_and_poll();
}

PollResult DeleteBlockJob::poll(Worker& worker) {
    auto buffer_id_opt = worker.m_block_manager.delete_block(block_id);

    if (buffer_id_opt) {
        worker.m_memory_manager->delete_buffer(*buffer_id_opt);
    }

    return PollResult::Ready;
}

PollResult EmptyJob::poll(Worker& worker) {
    return PollResult::Ready;
}

void Worker::WorkQueue::push(std::shared_ptr<WorkerJob> op) const {
    std::lock_guard guard {m_lock};

    if (!op->is_ready) {
        bool was_empty = m_queue.empty();

        op->is_ready = true;
        m_queue.push_back(std::move(op));

        // If this is the first entry, notify any waiters
        if (was_empty) {
            m_cond.notify_one();
        }
    }
}

void Worker::WorkQueue::pop_nonblocking(std::deque<std::shared_ptr<WorkerJob>>& output) const {
    std::unique_lock guard {m_lock};
    pop_impl(output);
}

void Worker::WorkQueue::pop_blocking(
    std::chrono::time_point<std::chrono::system_clock> deadline,
    std::deque<std::shared_ptr<WorkerJob>>& output) const {
    std::unique_lock guard {m_lock};
    while (m_queue.empty()) {
        if (m_cond.wait_until(guard, deadline) == std::cv_status::timeout) {
            return;
        }
    }

    pop_impl(output);
}

void Worker::WorkQueue::pop_impl(std::deque<std::shared_ptr<WorkerJob>>& output) const {
    while (!m_queue.empty()) {
        auto op = std::move(m_queue.front());
        m_queue.pop_front();

        op->is_ready = false;

        output.push_back(std::move(op));
    }
}

Worker::Worker(
    std::vector<std::shared_ptr<Executor>> executors,
    std::unique_ptr<MemoryManager> memory_manager) :
    m_executors(std::move(executors)),
    m_memory_manager(std::move(memory_manager)) {}

void Worker::make_progress(std::chrono::time_point<std::chrono::system_clock> deadline) {
    make_progress_impl(std::unique_lock {m_lock}, deadline);
}

void Worker::wakeup(std::shared_ptr<WorkerJob> op, bool allow_progress) {
    if (allow_progress) {
        if (auto guard = std::unique_lock {m_lock, std::try_to_lock}) {
            m_queue.push_back(std::move(op));
            make_progress_impl(std::move(guard));
            return;
        }
    }

    m_poll_queue.push(op);
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
        op = std::make_shared<DeleteBlockJob>(id, cmd->id);
    } else if (std::get_if<CommandNoop>(&command)) {
        op = std::make_shared<EmptyJob>(id);
    } else {
        KMM_PANIC("invalid command");
    }

    m_scheduler.insert_job(std::move(op), std::move(dependencies));
    make_progress_impl(std::move(guard));
}

bool Worker::query_event(EventId id, std::chrono::time_point<std::chrono::system_clock> deadline) {
    std::unique_lock<std::mutex> guard {m_lock};
    return m_scheduler.is_job_complete_with_deadline(id, guard, deadline);
}

void Worker::shutdown() {
    std::unique_lock<std::mutex> guard {m_lock};
    m_shutdown = true;
}

bool Worker::has_shutdown() {
    std::unique_lock<std::mutex> guard {m_lock};
    return m_shutdown && m_scheduler.all_complete();
}

void Worker::make_progress_impl(
    std::unique_lock<std::mutex> guard,
    std::chrono::time_point<std::chrono::system_clock> deadline) {
    m_poll_queue.pop_nonblocking(m_queue);
    if (m_queue.empty() && deadline != std::chrono::time_point<std::chrono::system_clock>()) {
        guard.unlock();
        m_poll_queue.pop_blocking(deadline, m_queue);
        guard.lock();
    }

    while (true) {
        if (!m_queue.empty()) {
            for (const auto& op : m_queue) {
                poll_job(op);
            }

            m_poll_queue.pop_nonblocking(m_queue);
            continue;
        }

        if (auto ready = m_scheduler.pop_ready_job()) {
            auto op = std::dynamic_pointer_cast<WorkerJob>(*ready);
            KMM_ASSERT(op != nullptr);

            op->is_running = true;
            op->start(*this);

            poll_job(op);
            continue;
        }

        break;
    }
}

void Worker::poll_job(const std::shared_ptr<WorkerJob>& op) {
    if (op->is_running && op->poll(*this) == PollResult::Ready) {
        spdlog::debug("scheduling operation id={}", op->id());
        op->is_running = false;
        op->stop(*this);
        m_scheduler.mark_job_complete(op->id());
    }
}

}  // namespace kmm

#include <functional>
#include <stdexcept>
#include <utility>

#include "fmt/ranges.h"
#include "spdlog/spdlog.h"

#include "kmm/memory_manager.hpp"
#include "kmm/scheduler.hpp"
#include "kmm/utils.hpp"

namespace kmm {

struct Scheduler::Operation:
    public Waker,
    TaskCompletion::Impl,
    std::enable_shared_from_this<Operation> {
    enum class Status { Pending, Queueing, Staging, Running, Done };

    Operation(EventId id, Command kind, size_t unsatisfied, std::weak_ptr<Scheduler> scheduler) :
        id(id),
        command(std::move(kind)),
        unsatisfied_predecessors(unsatisfied),
        scheduler(std::move(scheduler)) {}

    EventId id;
    Status status = Status::Pending;
    Command command;
    size_t unsatisfied_predecessors = 1;
    std::vector<std::shared_ptr<Operation>> successors = {};
    std::vector<MemoryRequest> memory_requests = {};

    std::weak_ptr<Scheduler> scheduler;
    bool is_ready = false;

    void wakeup() const override {
        if (auto s = scheduler.lock()) {
            s->wakeup(const_cast<Operation*>(this)->shared_from_this());
        }
    }

    void complete_task(TaskResult result) override {
        std::shared_ptr(scheduler)->complete(shared_from_this(), std::move(result));
    }
};

Scheduler::Scheduler(
    std::vector<std::shared_ptr<Executor>> executors,
    std::shared_ptr<MemoryManager> memory_manager,
    std::shared_ptr<BlockManager> block_manager) :
    m_executors(std::move(executors)),
    m_memory_manager(std::move(memory_manager)),
    m_block_manager(std::move(block_manager)) {}

void Scheduler::make_progress(
    std::optional<std::chrono::time_point<std::chrono::system_clock>> deadline) {
    make_progress_impl(std::unique_lock {m_lock}, deadline);
}

void Scheduler::complete(const std::shared_ptr<Operation>& op, TaskResult result) {
    std::unique_lock guard {m_lock};
    complete_op(op, std::move(result));
    make_progress_impl(std::move(guard));
}

void Scheduler::wakeup(const std::shared_ptr<Operation>& op) {
    m_ready_queue.push(op);
}

void Scheduler::submit_command(EventId id, Command command, EventList dependencies) {
    std::unique_lock guard {m_lock};

    spdlog::debug("submit command id={} dependencies={} command={}", id, dependencies, command);

    if (m_shutdown) {
        throw std::runtime_error("cannot submit new commands after shutdown");
    }

    dependencies.remove_duplicates();

    auto new_op = std::make_shared<Operation>(
        id,
        std::move(command),
        dependencies.size() + 1,
        weak_from_this());
    m_ops.insert({id, new_op});

    size_t satisfied = 1;
    // auto predecessors = std::vector<std::weak_ptr<Operation>> {};

    for (auto dep_id : dependencies) {
        auto it = m_ops.find(dep_id);
        if (it == m_ops.end()) {
            satisfied++;
            continue;
        }

        auto& predecessor = it->second;
        predecessor->successors.push_back(new_op);
        // predecessors.push_back(predecessor);
    }

    // We always add one "phantom" predecessor to `predecessors_pending` so we can trigger it here
    trigger_predecessor_completed(new_op, satisfied);
}

bool Scheduler::query_event(
    EventId id,
    std::chrono::time_point<std::chrono::system_clock> deadline) {
    std::unique_lock<std::mutex> guard {m_lock};

    while (true) {
        if (m_ops.find(id) == m_ops.end()) {
            return true;
        }

        if (deadline == std::chrono::time_point<std::chrono::system_clock>()) {
            return false;
        }

        if (m_ops_waiters.wait_until(guard, deadline) == std::cv_status::timeout) {
            return false;
        }
    }
}

void Scheduler::shutdown() {
    std::unique_lock<std::mutex> guard {m_lock};
    m_shutdown = true;
}

bool Scheduler::has_shutdown() {
    std::unique_lock<std::mutex> guard {m_lock};
    return m_shutdown && m_ops.empty();
}

void Scheduler::ReadyQueue::push(std::shared_ptr<Operation> op) const {
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

void Scheduler::ReadyQueue::pop_nonblocking(std::deque<std::shared_ptr<Operation>>& output) const {
    std::unique_lock guard {m_lock};
    pop_impl(output);
}

void Scheduler::ReadyQueue::pop_blocking(
    std::chrono::time_point<std::chrono::system_clock> deadline,
    std::deque<std::shared_ptr<Operation>>& output) const {
    std::unique_lock guard {m_lock};
    while (m_queue.empty()) {
        if (m_cond.wait_until(guard, deadline) == std::cv_status::timeout) {
            return;
        }
    }

    pop_impl(output);
}

void Scheduler::ReadyQueue::pop_impl(std::deque<std::shared_ptr<Operation>>& output) const {
    while (!m_queue.empty()) {
        auto op = std::move(m_queue.front());
        m_queue.pop_front();

        op->is_ready = false;

        output.push_back(std::move(op));
    }
}

void Scheduler::make_progress_impl(
    std::unique_lock<std::mutex> guard,
    std::optional<std::chrono::time_point<std::chrono::system_clock>> deadline) {
    std::deque<std::shared_ptr<Operation>> local_ready;

    while (true) {
        m_ready_queue.pop_nonblocking(local_ready);

        // Empty? Try waiting for an operation without holding the lock.
        if (local_ready.empty() && deadline) {
            guard.unlock();
            m_ready_queue.pop_blocking(*deadline, local_ready);
            guard.lock();
        }

        // Still empty? That means the deadline has been exceeded in `pop_blocking`.
        if (local_ready.empty()) {
            break;
        }

        while (!local_ready.empty()) {
            auto op = std::move(local_ready.front());
            local_ready.pop_front();

            poll_op(op);
        }
    }
}

void Scheduler::poll_op(const std::shared_ptr<Operation>& op) {
    if (op->status == Operation::Status::Pending) {
        if (op->unsatisfied_predecessors == 0) {
            stage_op(op);
        }
    } else if (op->status == Operation::Status::Staging) {
        if (m_memory_manager->poll_requests(op->memory_requests) == PollResult::Ready) {
            schedule_op(op);
        }
    }
}

void Scheduler::stage_op(const std::shared_ptr<Operation>& op) {
    op->status = Operation::Status::Staging;
    bool is_ready = true;

    if (const auto& cmd_exe = std::get_if<CommandExecute>(&op->command)) {
        auto requests = std::vector<MemoryRequest>();

        for (const auto& arg : cmd_exe->inputs) {
            auto buffer_id_opt = m_block_manager->get_block_buffer(arg.block_id);

            if (buffer_id_opt) {
                requests.push_back(m_memory_manager->create_request(  //
                    *buffer_id_opt,
                    arg.memory_id,
                    false,
                    op));
            }
        }

        for (const auto& arg : cmd_exe->outputs) {
            auto layout = arg.meta->layout();

            if (layout.num_bytes > 0) {
                auto buffer_id = m_memory_manager->create_buffer(arg.meta->layout());
                requests.push_back(m_memory_manager->create_request(  //
                    buffer_id,
                    arg.memory_id,
                    true,
                    op));
            }
        }

        is_ready = m_memory_manager->poll_requests(requests) == PollResult::Ready;
        op->memory_requests = std::move(requests);
    }

    if (is_ready) {
        schedule_op(op);
    }
}

void Scheduler::schedule_op(const std::shared_ptr<Operation>& op) {
    spdlog::debug("scheduling operation id={}", op->id);

    op->status = Operation::Status::Running;
    bool is_done = true;

    if (const auto* cmd_exe = std::get_if<CommandExecute>(&op->command)) {
        auto context = TaskContext {};
        size_t index = 0;

        for (const auto& input : cmd_exe->inputs) {
            const auto& req = op->memory_requests[index++];
            const auto* allocation = m_memory_manager->view_buffer(req);
            auto header = m_block_manager->get_block_header(input.block_id);

            context.inputs.push_back(InputBlock {
                .block_id = input.block_id,
                .header = header,
                .allocation = allocation,
            });
        }

        for (const auto& output : cmd_exe->outputs) {
            const auto& req = op->memory_requests[index++];
            const auto* allocation = m_memory_manager->view_buffer(req);

            context.outputs.push_back(OutputBuffer {
                .block_id = output.block_id,
                .header = output.meta.get(),
                .allocation = allocation,
            });
        }

        is_done = false;
        m_executors.at(cmd_exe->device_id)
            ->submit(cmd_exe->task, std::move(context), TaskCompletion(op));

    } else if (const auto* cmd_noop = std::get_if<CommandNoop>(&op->command)) {
        // Nothing to do

    } else if (const auto* cmd_del = std::get_if<CommandBlockDelete>(&op->command)) {
        auto buffer_id_opt = m_block_manager->delete_block(cmd_del->id);

        if (buffer_id_opt) {
            m_memory_manager->delete_buffer(*buffer_id_opt);
        }
    } else {
        throw std::runtime_error("invalid command");
    }

    if (is_done) {
        complete_op(op, {});
    }
}

void Scheduler::complete_op(const std::shared_ptr<Operation>& op, TaskResult result) {
    KMM_ASSERT(op->status == Operation::Status::Running);
    spdlog::debug("completing operation={} result={}", op->id, result.index());
    op->status = Operation::Status::Done;

    for (const auto& request : op->memory_requests) {
        m_memory_manager->delete_request(request);
    }
    op->memory_requests.clear();

    if (auto* cmd_exe = std::get_if<CommandExecute>(&op->command)) {
        size_t num_outputs = cmd_exe->outputs.size();
        const auto* error = std::get_if<TaskError>(&result);

        for (size_t i = 0; i < num_outputs; i++) {
            auto& output = cmd_exe->outputs[i];
            auto block_id = output.block_id;
            auto buffer_id = BufferId::invalid();  // TODO

            if (error == nullptr) {
                m_block_manager->insert_block(block_id, std::move(output.meta), buffer_id);
            } else {
                m_block_manager->poison_block(block_id, *error);
                m_memory_manager->delete_buffer(buffer_id);
            }
        }
    }

    for (const auto& successor : op->successors) {
        trigger_predecessor_completed(successor);
    }

    m_ops.erase(op->id);
    m_ops_waiters.notify_all();
}

void Scheduler::trigger_predecessor_completed(const std::shared_ptr<Operation>& op, size_t count) {
    spdlog::debug(
        "trigger for operation={} count={} unsatisfied={}",
        op->id,
        count,
        op->unsatisfied_predecessors);

    if (op->unsatisfied_predecessors <= count) {
        op->unsatisfied_predecessors = 0;
        m_ready_queue.push(op);
    } else {
        op->unsatisfied_predecessors -= count;
    }
}

}  // namespace kmm

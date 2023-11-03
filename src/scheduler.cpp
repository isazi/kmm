#include "kmm/scheduler.hpp"

#include <functional>
#include <stdexcept>
#include <utility>

#include "fmt/ranges.h"
#include "kmm/memory_manager.hpp"
#include "kmm/utils.hpp"
#include "spdlog/spdlog.h"

namespace kmm {

struct Scheduler::Operation: public Waker, std::enable_shared_from_this<Operation> {
    enum class Status { Pending, Queueing, Staging, Running, Done };

    Operation(
        OperationId id,
        Command kind,
        size_t unsatisfied,
        std::weak_ptr<Scheduler> scheduler) :
        id(id),
        command(std::move(kind)),
        unsatisfied_predecessors(unsatisfied),
        scheduler(std::move(scheduler)) {}

    OperationId id;
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
};

Scheduler::Scheduler(
    std::vector<std::shared_ptr<Executor>> executors,
    std::shared_ptr<MemoryManager> memory_manager,
    std::shared_ptr<ObjectManager> object_manager) :
    m_executors(std::move(executors)),
    m_memory_manager(std::move(memory_manager)),
    m_object_manager(std::move(object_manager)) {}

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

void Scheduler::submit_command(CommandPacket packet) {
    std::unique_lock guard {m_lock};

    spdlog::debug(
        "submit command id={} dependencies={} command={}",
        packet.id,
        packet.dependencies,
        packet.command);

    if (m_shutdown) {
        throw std::runtime_error("cannot submit new commands after shutdown");
    }

    remove_duplicates(packet.dependencies);

    auto new_op = std::make_shared<Operation>(
        packet.id,
        std::move(packet.command),
        packet.dependencies.size() + 1,
        weak_from_this());

    size_t satisfied = 1;
    // auto predecessors = std::vector<std::weak_ptr<Operation>> {};

    for (auto dep_id : packet.dependencies) {
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
        auto requests = std::vector<MemoryRequest> {};

        for (const auto& arg : cmd_exe->buffers) {
            requests.push_back(m_memory_manager->create_request(  //
                arg.buffer_id,
                arg.memory_id,
                arg.is_write,
                op));
        }

        is_ready = m_memory_manager->poll_requests(requests) == PollResult::Ready;
        op->memory_requests = std::move(requests);
    }

    if (is_ready) {
        schedule_op(op);
    }
}

void Scheduler::schedule_op(const std::shared_ptr<Operation>& op) {
    op->status = Operation::Status::Running;
    bool is_done = true;

    if (const auto* cmd_exe = std::get_if<CommandExecute>(&op->command)) {
        auto context = TaskContext {};

        for (auto& req : op->memory_requests) {
            const auto* allocation = m_memory_manager->view_buffer(req);
            context.buffers.push_back(BufferAccess {
                .allocation = allocation,
                .writable = false,
            });
        }

        is_done = false;
        m_executors.at(cmd_exe->device_id)
            ->submit(cmd_exe->task, std::move(context), TaskCompletion(op));

    } else if (const auto* cmd_noop = std::get_if<CommandNoop>(&op->command)) {
        // Nothing to do

    } else if (const auto* cmd_fut = std::get_if<CommandPromise>(&op->command)) {
        cmd_fut->promise.set_value();

    } else if (const auto* cmd_create = std::get_if<CommandBufferCreate>(&op->command)) {
        m_memory_manager->create_buffer(cmd_create->id, cmd_create->description);

    } else if (const auto* cmd_delete = std::get_if<CommandBufferDelete>(&op->command)) {
        m_memory_manager->delete_buffer(cmd_delete->id);

    } else if (const auto* cmd_new_obj = std::get_if<CommandObjectCreate>(&op->command)) {
        m_object_manager->create_object(cmd_new_obj->id, cmd_new_obj->object);

    } else if (const auto* cmd_del_obj = std::get_if<CommandObjectDelete>(&op->command)) {
        m_object_manager->delete_object(cmd_del_obj->id);

    } else {
        throw std::runtime_error("invalid command");
    }

    if (is_done) {
        complete_op(op, {});
    }
}

void Scheduler::complete_op(const std::shared_ptr<Operation>& op, TaskResult result) {
    op->status = Operation::Status::Done;
    m_ops.erase(op->id);

    if (const auto* cmd_exe = std::get_if<CommandExecute>(&op->command)) {
        if (auto output_id = cmd_exe->output_object_id) {
            if (const auto* obj = std::get_if<ObjectHandle>(&result)) {
                m_object_manager->create_object(*output_id, *obj);
            } else if (const auto* err = std::get_if<TaskError>(&result)) {
                m_object_manager->poison_object(*output_id, *err);
            } else {
                m_object_manager->poison_object(*output_id, TaskError("no output provided"));
            }
        }
    }

    for (const auto& request : op->memory_requests) {
        m_memory_manager->delete_request(request);
    }

    op->memory_requests.clear();

    for (const auto& successor : op->successors) {
        trigger_predecessor_completed(successor);
    }
}

void Scheduler::trigger_predecessor_completed(const std::shared_ptr<Operation>& op, size_t count) {
    if (op->unsatisfied_predecessors <= count) {
        op->unsatisfied_predecessors = 0;
        m_ready_queue.push(op);
    } else {
        op->unsatisfied_predecessors -= count;
    }
}

TaskCompletion::TaskCompletion(std::shared_ptr<Scheduler::Operation> op) : m_op(std::move(op)) {}

void TaskCompletion::complete(TaskResult result) {
    if (auto op = std::exchange(m_op, {})) {
        std::shared_ptr(op->scheduler)->complete(op, std::move(result));
    }
}

void TaskCompletion::complete_err(const std::string& error) {
    complete(TaskError(error));
}

TaskCompletion::~TaskCompletion() {
    if (m_op) {
        complete_err("tasks was not completed properly");
    }
}

}  // namespace kmm

#include "kmm/scheduler.hpp"

#include <atomic>
#include <functional>
#include <stdexcept>
#include <utility>

#include "kmm/memory_manager.hpp"
#include "kmm/utils.hpp"

namespace kmm {
enum class OperationStatus { Pending, Queueing, Ready, Staging, Running, Done };

struct Scheduler::Operation {
    Operation(OperationId id, Command kind, size_t unsatisfied) :
        id(id),
        command(std::move(kind)),
        unsatisfied_predecessors(unsatisfied) {}

    OperationId id;
    OperationStatus status = OperationStatus::Pending;
    Command command;
    size_t unsatisfied_predecessors = 1;
    std::vector<std::weak_ptr<Operation>> predecessors = {};
    std::vector<std::shared_ptr<Operation>> successors = {};
    std::vector<MemoryRequest> memory_requests = {};

    bool is_ready = false;
};

struct WakerImpl: public Waker {
    WakerImpl(std::shared_ptr<Scheduler> scheduler, std::shared_ptr<Scheduler::Operation> op) :
        scheduler(std::move(scheduler)),
        op(std::move(op)) {}

    void wakeup() const override {
        scheduler->wakeup(op);
    }

    std::shared_ptr<Scheduler> scheduler;
    std::shared_ptr<Scheduler::Operation> op;
};

void Scheduler::make_progress(std::chrono::time_point<std::chrono::system_clock> deadline) {
    make_progress_impl(deadline, std::unique_lock {m_lock});
}

void Scheduler::wakeup(const std::shared_ptr<Operation>& op) {
    m_ready_queue.push(op);
}

void Scheduler::complete(const std::shared_ptr<Operation>& op) {
    std::unique_lock guard {m_lock};
    complete_op(op);
    make_progress_impl(std::chrono::system_clock::now(), std::move(guard));
}

void Scheduler::submit_command(CommandPacket packet) {
    std::unique_lock guard {m_lock};
    remove_duplicates(packet.dependencies);

    auto new_op = std::make_shared<Operation>(
        packet.id,
        std::move(packet.command),
        packet.dependencies.size() + 1);

    auto predecessors = std::vector<std::weak_ptr<Operation>> {};
    size_t satisfied = 1;

    for (auto dep_id : packet.dependencies) {
        auto it = m_ops.find(dep_id);
        if (it == m_ops.end()) {
            satisfied++;
            continue;
        }

        auto& predecessor = it->second;
        predecessor->successors.push_back(new_op);
        predecessors.push_back(predecessor);
    }

    new_op->predecessors = std::move(predecessors);

    // We always add one "phantom" predecessor to `predecessors_pending` so we can trigger it here
    trigger_predecessor_completed(new_op, satisfied);
}

void Scheduler::ReadyQueue::push(const std::shared_ptr<Operation>& op) const {
    std::lock_guard guard {m_lock};
    if (!op->is_ready) {
        op->is_ready = true;
        m_queue.push_back(op);
        m_cond.notify_all();
    }
}

void Scheduler::ReadyQueue::pop_all(
    std::chrono::time_point<std::chrono::system_clock> deadline,
    std::deque<std::shared_ptr<Operation>>& output) const {
    std::unique_lock guard {m_lock};
    while (m_queue.empty()) {
        if (m_cond.wait_until(guard, deadline) == std::cv_status::timeout) {
            return;
        }
    }

    while (!m_queue.empty()) {
        auto op = std::move(m_queue.front());
        m_queue.pop_front();

        op->is_ready = false;

        output.push_back(std::move(op));
    }
}

void Scheduler::make_progress_impl(
    std::chrono::time_point<std::chrono::system_clock> deadline,
    std::unique_lock<std::mutex> guard) {
    std::deque<std::shared_ptr<Operation>> local_ready;

    do {
        guard.unlock();
        m_ready_queue.pop_all(deadline, local_ready);
        guard.lock();

        while (!local_ready.empty()) {
            poll_op(local_ready.front());
            local_ready.pop_front();
        }
    } while (std::chrono::system_clock::now() < deadline);
}

void Scheduler::stage_op(const std::shared_ptr<Operation>& op) {
    op->status = OperationStatus::Staging;
    bool is_ready = true;

    if (const auto& cmd_exe = std::get_if<CommandExecute>(&op->command)) {
        auto waker = std::make_shared<WakerImpl>(shared_from_this(), op);
        auto requests = std::vector<MemoryRequest> {};

        for (const auto& arg : cmd_exe->buffers) {
            requests.push_back(m_memory_manager->create_request(  //
                arg.buffer_id,
                arg.memory_id,
                arg.is_write,
                waker));
        }

        is_ready = m_memory_manager->poll_requests(requests) == PollResult::Ready;
        op->memory_requests = std::move(requests);
    }

    if (is_ready) {
        schedule_op(op);
    }
}

void Scheduler::poll_op(const std::shared_ptr<Operation>& op) {
    if (op->status == OperationStatus::Staging) {
        if (m_memory_manager->poll_requests(op->memory_requests) == PollResult::Ready) {
            schedule_op(op);
        }
    } else if (op->status == OperationStatus::Queueing && op->unsatisfied_predecessors == 0) {
        stage_op(op);
    }
}

void Scheduler::schedule_op(const std::shared_ptr<Operation>& op) {
    op->status = OperationStatus::Running;
    bool is_done = true;

    if (const auto* cmd_exe = std::get_if<CommandExecute>(&op->command)) {
        auto accessors = std::vector<BufferAccess> {};

        for (auto& req : op->memory_requests) {
            const auto* allocation = m_memory_manager->view_buffer(req);
            accessors.push_back(BufferAccess {
                .allocation = allocation,
                .writable = false,
            });
        }

        is_done = false;
        m_executors.at(cmd_exe->device_id)
            ->submit(cmd_exe->task, std::move(accessors), TaskCompletion(shared_from_this(), op));

    } else if (const auto* cmd_noop = std::get_if<CommandNoop>(&op->command)) {
        // Nothing to do

    } else if (const auto* cmd_fut = std::get_if<CommandPromise>(&op->command)) {
        cmd_fut->promise.set_value();

    } else if (const auto* cmd_create = std::get_if<CommandBufferCreate>(&op->command)) {
        m_memory_manager->create_buffer(cmd_create->id, cmd_create->description);

    } else if (const auto* cmd_delete = std::get_if<CommandBufferDelete>(&op->command)) {
        m_memory_manager->delete_buffer(cmd_delete->id);

    } else if (const auto* cmd_obj = std::get_if<CommandObjectDelete>(&op->command)) {
        m_object_manager->delete_object(cmd_obj->id);

    } else {
        throw std::runtime_error("invalid command");
    }

    if (is_done) {
        complete_op(op);
    }
}

void Scheduler::complete_op(const std::shared_ptr<Operation>& op) {
    op->status = OperationStatus::Done;
    m_ops.erase(op->id);

    for (const auto& successor : op->successors) {
        trigger_predecessor_completed(successor);
    }
}

void Scheduler::trigger_predecessor_completed(const std::shared_ptr<Operation>& op, size_t count) {
    if (op->unsatisfied_predecessors <= count) {
        op->unsatisfied_predecessors = 0;
        op->status = OperationStatus::Queueing;
        m_ready_queue.push(op);
    } else {
        op->unsatisfied_predecessors -= count;
    }
}

TaskCompletion::TaskCompletion(
    std::weak_ptr<Scheduler> worker,
    std::weak_ptr<Scheduler::Operation> task) :
    m_scheduler(std::move(worker)),
    m_op(std::move(task)) {}

void TaskCompletion::complete() {
    if (auto op = m_op.lock()) {
        m_op.reset();
        m_scheduler.lock()->complete(op);
    }
}

TaskCompletion::~TaskCompletion() {
    complete();
}

}  // namespace kmm

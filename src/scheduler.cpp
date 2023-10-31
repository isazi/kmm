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
    std::atomic<bool> is_ready = false;
};

struct WakerImpl: public Waker {
    WakerImpl(std::shared_ptr<Scheduler> scheduler, std::shared_ptr<Scheduler::Operation> op) :
        scheduler(std::move(scheduler)),
        op(std::move(op)) {}

    void wakeup() const override {
        scheduler->wakeup_op(op);
    }

    std::shared_ptr<Scheduler> scheduler;
    std::shared_ptr<Scheduler::Operation> op;
};

void Scheduler::make_progress(std::chrono::time_point<std::chrono::system_clock> deadline) {
    while (true) {
        std::unique_lock guard {m_ready_lock};
        bool not_empty =
            m_ready_cond.wait_until(guard, deadline, [&] { return !m_ready_ops.empty(); });

        if (!not_empty) {
            break;
        }

        auto op = std::move(m_ready_ops.front());
        m_ready_ops.pop_front();
        guard.unlock();

        poll_op(op);
    }
}

void Scheduler::submit_command(CommandPacket packet) {
    remove_duplicates(packet.dependencies);

    auto new_op = std::make_shared<Operation>(
        packet.id,
        std::move(packet.command),
        packet.dependencies.size() + 1);

    auto predecessors = std::vector<std::weak_ptr<Operation>> {};
    predecessors.reserve(packet.dependencies.size());
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

void Scheduler::wakeup_op(const std::shared_ptr<Operation>& op) {
    if (op->is_ready.exchange(true) == false) {
        {
            std::lock_guard guard {m_ready_lock};
            m_ready_ops.push_back(op);
        }

        m_ready_cond.notify_all();
    }
}

void Scheduler::trigger_predecessor_completed(const std::shared_ptr<Operation>& op, size_t count) {
    if (op->unsatisfied_predecessors < count) {
        op->unsatisfied_predecessors = 0;
        op->status = OperationStatus::Queueing;

        std::lock_guard guard {m_ready_lock};
        m_ready_ops.push_back(op);
    } else {
        op->unsatisfied_predecessors -= count;
    }
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
    op->is_ready.store(false);

    if (op->status == OperationStatus::Staging) {
        if (m_memory_manager->poll_requests(op->memory_requests) == PollResult::Ready) {
            schedule_op(op);
        }
    } else if (op->status == OperationStatus::Queueing) {
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
    // TODO: Get lock
    op->status = OperationStatus::Done;
    m_ops.erase(op->id);

    for (const auto& successor : op->successors) {
        trigger_predecessor_completed(successor);
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
        m_scheduler.lock()->complete_op(op);
    }
}

TaskCompletion::~TaskCompletion() {
    complete();
}

}  // namespace kmm

#include "kmm/scheduler.hpp"

#include <functional>
#include <stdexcept>
#include <utility>

#include "kmm/memory_manager.hpp"
#include "kmm/utils.hpp"

namespace kmm {
struct Scheduler::Job {
    Job(JobId, Command);

    JobId id;
    Command command;
    JobStatus status = JobStatus::Pending;
    size_t requests_pending = 0;
    std::vector<MemoryRequest> requests = {};
    size_t predecessors_pending = 1;
    std::vector<std::weak_ptr<Job>> predecessors = {};
    std::vector<std::shared_ptr<Job>> successors = {};
};

void Scheduler::make_progress() {
    // TODO
}

void Scheduler::trigger_predecessor_completed(const std::shared_ptr<Job>& job) {
    job->predecessors_pending -= 1;

    if (job->predecessors_pending == 0) {
        job->status = JobStatus::Ready;
        m_ready_tasks.push_back(job);
    }
}

struct Scheduler::JobWaker: public Waker {
    JobWaker(std::shared_ptr<Job> job, std::shared_ptr<Scheduler> scheduler) :
        m_job(std::move(job)),
        m_scheduler(std::move(scheduler)) {}

    void wakeup() const {
        KMM_TODO();
    }

    std::shared_ptr<Job> m_job;
    std::shared_ptr<Scheduler> m_scheduler;
};

void Scheduler::stage_job(const std::shared_ptr<Job>& job) {
    auto requests = std::vector<MemoryRequest> {};

    if (const auto* cmd_exe = std::get_if<CommandExecute>(&(job->command))) {
        auto waker = std::make_shared<JobWaker>(job, shared_from_this());

        for (const auto& arg : cmd_exe->buffers) {
            requests.push_back(m_memory_manager->create_request(  //
                arg.buffer_id,
                arg.memory_id,
                arg.is_write,
                waker));
        }
    }

    size_t requests_pending = requests.size();
    job->status = JobStatus::Staging;
    job->requests_pending = requests_pending;
    job->requests = std::move(requests);

    if (requests_pending == 0) {
        schedule_job(job);
    }
}

void Scheduler::schedule_job(const std::shared_ptr<Job>& job) {
    job->status = JobStatus::Scheduled;
    bool is_done = true;

    if (const auto* cmd_exe = std::get_if<CommandExecute>(&(job->command))) {
        is_done = false;

        std::vector<BufferAccess> allocations;
        for (const auto& req : job->requests) {
            allocations.push_back(BufferAccess {
                .allocation = m_memory_manager->view_buffer(req),
                .writable = false,
            });
        }

        auto context = TaskCompletion(shared_from_this(), job);
        m_executors.at(cmd_exe->device_id)
            ->submit(
                cmd_exe->task,  //
                std::move(allocations),
                std::move(context));

    } else if (const auto* cmd_noop = std::get_if<CommandNoop>(&job->command)) {
        // Nothing to do

    } else if (const auto* cmd_create = std::get_if<CommandBufferCreate>(&job->command)) {
        m_memory_manager->create_buffer(cmd_create->id, cmd_create->description);

    } else if (const auto* cmd_delete = std::get_if<CommandBufferDelete>(&job->command)) {
        m_memory_manager->delete_buffer(cmd_delete->id);

    } else if (const auto* cmd_obj = std::get_if<CommandObjectDelete>(&job->command)) {
        m_object_manager->delete_object(cmd_obj->id);

    } else {
        throw std::runtime_error("invalid job command");
    }

    if (is_done) {
        complete_job(job);
    }
}

void Scheduler::complete_job(const std::shared_ptr<Job>& job) {
    job->status = JobStatus::Done;
    m_jobs.erase(job->id);

    for (const auto& request : job->requests) {
        m_memory_manager->delete_request(request, std::nullopt);
    }

    for (const auto& successor : job->successors) {
        trigger_predecessor_completed(successor);
    }
}

void Scheduler::submit_command(CommandPacket packet) {
    auto new_job = std::make_shared<Job>(packet.id, std::move(packet.command));

    remove_duplicates(packet.dependencies);

    auto predecessors = std::vector<std::weak_ptr<Job>> {};
    predecessors.reserve(packet.dependencies.size());

    for (auto dep_id : packet.dependencies) {
        auto it = m_jobs.find(dep_id);
        if (it == m_jobs.end()) {
            continue;
        }

        auto& predecessor = it->second;
        predecessor->successors.push_back(new_job);
        predecessors.push_back(predecessor);
    }

    size_t predecessors_pending = predecessors.size();
    new_job->predecessors = std::move(predecessors);
    new_job->predecessors_pending = predecessors_pending + 1;

    // We always add one "phantom" predecessor to `predecessors_pending` so we can trigger it here
    trigger_predecessor_completed(new_job);
}

Scheduler::Job::Job(JobId id, Command kind) :
    id(id),
    command(std::move(kind)),
    status(Scheduler::JobStatus::Pending) {}

TaskCompletion::TaskCompletion(
    std::weak_ptr<Scheduler> worker,
    std::weak_ptr<Scheduler::Job> task) :
    m_worker(std::move(worker)),
    m_job(std::move(task)) {}

void TaskCompletion::complete() {
    if (auto job = m_job.lock()) {
        m_job.reset();
        m_worker.lock()->complete_job(job);
    }
}

TaskCompletion::~TaskCompletion() {
    complete();
}

}  // namespace kmm

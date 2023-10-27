#include "kmm/worker.hpp"

#include <functional>
#include <stdexcept>
#include <utility>

#include "kmm/memory_manager.hpp"
#include "kmm/utils.hpp"

namespace kmm {
void Worker::make_progress() {
    while (true) {
        if (auto req = m_memory_manager.poll()) {
            auto job = std::dynamic_pointer_cast<Job>(*req);
            job->requests_pending -= 1;

            if (job->requests_pending == 0) {
                schedule_task(job);
            }
            continue;
        }

        if (!m_ready_tasks.empty()) {
            auto task = std::move(m_ready_tasks.front());
            m_ready_tasks.pop_front();

            stage_task(task);
            continue;
        }

        break;
    }
}

void Worker::trigger_predecessor_completed(const std::shared_ptr<Job>& job) {
    job->predecessors_pending -= 1;

    if (job->predecessors_pending == 0) {
        job->status = JobStatus::Ready;
        m_ready_tasks.push_back(job);
    }
}

void Worker::stage_task(const std::shared_ptr<Job>& job) {
    auto requests = std::vector<std::shared_ptr<MemoryRequest>> {};

    for (const auto& arg : job->buffers) {
        requests.push_back(
            m_memory_manager
                .acquire_buffer_access(arg.buffer_id, arg.memory_id, arg.is_write, job));
    }

    size_t requests_pending = requests.size();
    job->status = JobStatus::Staging;
    job->requests_pending = requests_pending;
    job->requests = std::move(requests);

    if (requests_pending == 0) {
        schedule_task(job);
    }
}

void Worker::schedule_task(const std::shared_ptr<Job>& job) {
    job->status = JobStatus::Scheduled;

    if (const auto* cmd_exe = std::get_if<CommandExecute>(&(job->command))) {
        std::vector<std::shared_ptr<Allocation>> allocations;
        for (const auto& req : job->requests) {
            allocations.push_back(m_memory_manager.view_buffer(req));
        }

        cmd_exe->device_id;
        cmd_exe->task;
        // TODO
    } else if (const auto* cmd_noop = std::get_if<CommandNoop>(&job->command)) {
        complete_task(job);
    } else if (const auto* cmd_create = std::get_if<CommandBufferCreate>(&job->command)) {
        m_memory_manager.create_buffer(cmd_create->id, cmd_create->description);
        complete_task(job);
    } else if (const auto* cmd_delete = std::get_if<CommandBufferDelete>(&job->command)) {
        m_memory_manager.delete_buffer(cmd_create->id);
        complete_task(job);
    } else {
        throw std::runtime_error("invalid task kind");
    }
}

void Worker::complete_task(const std::shared_ptr<Job>& job) {
    job->status = JobStatus::Done;
    m_tasks.erase(job->id);

    for (const auto& request : job->requests) {
        m_memory_manager.release_buffer_access(request, std::nullopt);
    }

    for (const auto& successor : job->successors) {
        trigger_predecessor_completed(successor);
    }
}

void Worker::submit_command(
    JobId id,
    Command command,
    std::vector<JobId> dependencies,
    std::vector<BufferRequirement> buffers) {
    remove_duplicates(dependencies);

    auto predecessors = std::vector<std::weak_ptr<Job>> {};
    predecessors.reserve(dependencies.size());

    auto new_job = std::make_shared<Job>(id, std::move(command), std::move(buffers));

    for (auto dep_id : dependencies) {
        auto it = m_tasks.find(dep_id);
        if (it == m_tasks.end()) {
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

Worker::Job::Job(JobId id, Command kind, std::vector<BufferRequirement> buffers) :
    id(id),
    command(std::move(kind)),
    status(Worker::JobStatus::Pending),
    buffers(std::move(buffers)) {}

TaskContext::TaskContext(
    std::shared_ptr<Worker> worker,
    std::shared_ptr<Worker::Job> task,
    std::vector<BufferAccess> buffers) :
    m_worker(std::move(worker)),
    m_task(std::move(task)),
    m_buffers(std::move(buffers)) {}

TaskContext::~TaskContext() {
    if (m_task) {
        m_worker->complete_task(m_task);
        m_task.reset();
    }
}

}  // namespace kmm

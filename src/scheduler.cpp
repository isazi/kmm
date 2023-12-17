#include "kmm/scheduler.hpp"

namespace kmm {

void Scheduler::insert_job(std::shared_ptr<Job> node, EventList dependencies) {
    KMM_ASSERT(node->status == Job::Status::Created);
    node->status = Job::Status::Pending;

    dependencies.remove_duplicates();
    node->unsatisfied_predecessors = dependencies.size();

    m_jobs.insert({node->identifier, node});

    size_t satisfied = 1;
    // auto predecessors = std::vector<std::weak_ptr<Operation>> {};

    for (auto dep_id : dependencies) {
        auto it = m_jobs.find(dep_id);

        if (it == m_jobs.end()) {
            satisfied++;
            continue;
        }

        auto predecessor = it->second;
        predecessor->successors.push_back(node);
        // predecessors.push_back(predecessor);
    }

    // We always add one "phantom" predecessor to `predecessors_pending` so we can trigger it here
    trigger_predecessor_completed(node, satisfied);
}

void Scheduler::mark_job_complete(EventId id) {
    auto it = m_jobs.find(id);
    if (it == m_jobs.end()) {
        return;
    }

    auto& op = it->second;

    KMM_ASSERT(op->status == Job::Status::Running);
    op->status = Job::Status::Done;

    for (const auto& successor : op->successors) {
        trigger_predecessor_completed(successor);
    }

    m_jobs.erase(it);
    m_completion_condvar.notify_all();
}

bool Scheduler::all_complete() const {
    return m_jobs.empty();
}

bool Scheduler::is_job_complete(EventId id) const {
    return m_jobs.find(id) == m_jobs.end();
}

bool Scheduler::is_job_complete_with_deadline(
    EventId id,
    std::unique_lock<std::mutex>& guard,
    std::chrono::time_point<std::chrono::system_clock> deadline) {
    while (true) {
        if (is_job_complete(id)) {
            return true;
        }

        if (deadline == std::chrono::time_point<std::chrono::system_clock>()) {
            return false;
        }

        if (m_completion_condvar.wait_until(guard, deadline) == std::cv_status::timeout) {
            return false;
        }
    }
}

std::optional<std::shared_ptr<Job>> Scheduler::pop_ready_job() {
    if (m_ready_queue.empty()) {
        return std::nullopt;
    }

    auto op = std::move(m_ready_queue.front());
    op->status = Job::Status::Running;
    m_ready_queue.pop_front();

    return op;
}

void Scheduler::trigger_predecessor_completed(const std::shared_ptr<Job>& op, size_t count) {
    spdlog::debug(
        "trigger for operation={} count={} unsatisfied={}",
        op->id(),
        count,
        op->unsatisfied_predecessors);

    if (op->unsatisfied_predecessors <= count) {
        op->unsatisfied_predecessors = 0;
        m_ready_queue.push_back(op);
    } else {
        op->unsatisfied_predecessors -= count;
    }
}

}  // namespace kmm
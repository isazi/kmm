#include "kmm/worker/scheduler.hpp"

namespace kmm {

std::shared_ptr<Scheduler::Node> Scheduler::insert(
    EventId id,
    Command command,
    EventList dependencies,
    std::shared_ptr<Node> parent) {
    if (parent) {
        KMM_ASSERT(parent->status == Node::Status::Running);
        parent->active_children++;
    }

    auto node =
        std::make_shared<Node>(id, dependencies.size() + 1, std::move(command), std::move(parent));
    m_jobs.insert({id, node});

    size_t satisfied = 1;

    for (auto dep_id : dependencies) {
        auto it = m_jobs.find(dep_id);

        if (it != m_jobs.end()) {
            auto predecessor = it->second;
            predecessor->successors.push_back(node);
        } else {
            satisfied++;
        }
    }

    satisfy_job_dependencies(node, satisfied);
    return node;
}

void Scheduler::push_ready(std::shared_ptr<Node> node) {
    if (std::holds_alternative<ExecuteCommand>(node->command)) {
        m_ready_tasks.push_back(std::move(node));
    } else {
        m_ready_aux.push_back(std::move(node));
    }
}

std::optional<std::shared_ptr<Scheduler::Node>> Scheduler::pop_ready() {
    std::shared_ptr<Scheduler::Node> node;

    if (!m_ready_aux.empty()) {
        node = std::move(m_ready_aux.front());
        m_ready_aux.pop_front();
    } else if (!m_ready_tasks.empty()) {
        node = std::move(m_ready_tasks.front());
        m_ready_tasks.pop_front();
    } else {
        return std::nullopt;
    }

    KMM_ASSERT(node->status == Node::Status::Ready);
    node->status = Node::Status::Running;
    return node;
}

void Scheduler::complete(std::shared_ptr<Node> node) {
    KMM_ASSERT(node->status == Node::Status::Running);
    node->status = Node::Status::WaitingForChildren;

    if (node->active_children == 0) {
        complete_impl(std::move(node));
    }
}

void Scheduler::complete_impl(std::shared_ptr<Node> node) {
    auto it = m_jobs.find(node->id());
    if (it == m_jobs.end() || it->second.get() != node.get()) {
        return;
    }

    m_jobs.erase(it);
    node->status = Node::Status::Done;

    auto successors = std::move(node->successors);
    for (auto& successor : successors) {
        satisfy_job_dependencies(std::move(successor));
    }

    if (auto parent = std::move(node->parent)) {
        parent->active_children--;

        if (parent->active_children == 0 && parent->status == Node::Status::WaitingForChildren) {
            complete_impl(std::move(parent));
        }
    }
}

void Scheduler::satisfy_job_dependencies(std::shared_ptr<Node> node, size_t satisfied) {
    KMM_ASSERT(node->status == Node::Status::Pending);

    if (node->unsatisfied_predecessors > satisfied) {
        node->unsatisfied_predecessors -= satisfied;
        return;
    }

    // Job is now ready!
    node->unsatisfied_predecessors = 0;
    node->status = Node::Status::Ready;
    push_ready(std::move(node));
}

bool Scheduler::is_all_completed() const {
    return m_jobs.empty();
}

bool Scheduler::is_completed(EventId id) const {
    return m_jobs.find(id) == m_jobs.end();
}

}  // namespace kmm
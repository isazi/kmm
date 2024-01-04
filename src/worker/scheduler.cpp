#include <algorithm>

#include "kmm/worker/scheduler.hpp"

namespace kmm {

Scheduler::Scheduler() {
    m_ready_queues.emplace_back(ReadyQueue {
        .current_workload = 0,
        .max_workload = std::numeric_limits<size_t>::max(),
        .ready = {}});
}

std::shared_ptr<Scheduler::Node> Scheduler::insert(
    EventId id,
    Command command,
    EventList dependencies,
    std::shared_ptr<Node> parent) {
    if (parent) {
        KMM_ASSERT(parent->status == Node::Status::Running);
        parent->active_children++;
    }

    std::sort(dependencies.begin(), dependencies.end());
    const EventId* unique_end = std::unique(dependencies.begin(), dependencies.end());
    auto num_dependencies = static_cast<size_t>(unique_end - dependencies.begin());

    auto node = std::make_shared<Node>(
        id,  //
        num_dependencies + 1,
        std::move(command),
        std::move(parent),
        m_next_sequence_number++);
    m_tasks.insert({id, node});

    size_t satisfied = 1;

    for (size_t i = 0; i < num_dependencies; i++) {
        auto it = m_tasks.find(dependencies[i]);

        if (it != m_tasks.end()) {
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
    while (node->queue_id >= m_ready_queues.size()) {
        m_ready_queues.emplace_back();
    }

    m_ready_queues[node->queue_id].ready.emplace(node->sequence_number, std::move(node));
}

std::optional<std::shared_ptr<Scheduler::Node>> Scheduler::pop_ready() {
    std::shared_ptr<Scheduler::Node> node;

    for (auto& q : m_ready_queues) {
        if (q.current_workload < q.max_workload) {
            auto it = q.ready.begin();
            node = std::move(it->second);
            q.ready.erase(it);
            break;
        }
    }

    if (!node) {
        return std::nullopt;
    }

    KMM_ASSERT(node->status == Node::Status::Ready);
    node->status = Node::Status::Running;
    m_ready_queues[node->queue_id].current_workload++;

    return node;
}

void Scheduler::complete(std::shared_ptr<Node> node) {
    KMM_ASSERT(node->status == Node::Status::Running);
    node->status = Node::Status::WaitingForChildren;
    m_ready_queues[node->queue_id].current_workload--;

    if (node->active_children == 0) {
        complete_impl(std::move(node));
    }
}

void Scheduler::complete_impl(std::shared_ptr<Node> node) {
    auto it = m_tasks.find(node->id());
    if (it == m_tasks.end() || it->second.get() != node.get()) {
        return;
    }

    m_tasks.erase(it);
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
    return m_tasks.empty();
}

bool Scheduler::is_completed(EventId id) const {
    return m_tasks.find(id) == m_tasks.end();
}

}  // namespace kmm
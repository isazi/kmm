#include "kmm/internals/scheduler.hpp"

namespace kmm {

void Scheduler::insert_event(EventId event_id, Command command, const EventList& deps) {
    auto node = std::make_shared<TaskNode>(event_id, std::move(command));
    size_t num_not_scheduled = 0;
    size_t num_not_completed = 0;
    small_vector<CudaEvent> dependency_events;

    for (EventId dep_id : deps) {
        auto it = m_events.find(dep_id);

        if (it == m_events.end()) {
            continue;
        }

        auto& dep = it->second;
        dep->successors.push_back(node);
        num_not_completed++;

        if (auto event = dep->cuda_event) {
            dependency_events.push_back(*event);
        } else {
            num_not_scheduled++;
        }
    }

    node->dependencies_not_completed = num_not_completed;
    node->dependencies_not_scheduled = num_not_scheduled;
    node->dependency_events = std::move(dependency_events);
    enqueue_if_ready(node);

    m_events.emplace(event_id, std::move(node));
}

std::optional<std::shared_ptr<TaskNode>> Scheduler::pop_ready() {
    std::shared_ptr<TaskNode> result;

    if (!m_ready_high_priority.empty()) {
        result = std::move(m_ready_high_priority.front());
        m_ready_high_priority.pop_front();
    } else if (!m_ready_low_priority.empty()) {
        result = std::move(m_ready_low_priority.front());
        m_ready_low_priority.pop_front();
    } else {
        return std::nullopt;
    }

    result->status = TaskNode::Status::Scheduled;
    return result;
}

void Scheduler::set_scheduled(std::shared_ptr<TaskNode> node, CudaEvent event) {
    KMM_ASSERT(node->status == TaskNode::Status::Scheduled);
    KMM_ASSERT(node->cuda_event == std::nullopt);
    node->cuda_event = event;

    for (const auto& succ : node->successors) {
        succ->dependency_events.push_back(event);
        succ->dependencies_not_scheduled -= 1;
        enqueue_if_ready(succ);
    }
}

void Scheduler::set_complete(std::shared_ptr<TaskNode> node) {
    KMM_ASSERT(node->status == TaskNode::Status::Scheduled);
    node->status = TaskNode::Status::Done;
    node->cuda_event = std::nullopt;
    m_events.erase(node->event_id);

    for (const auto& succ : node->successors) {
        succ->dependencies_not_completed -= 1;
        enqueue_if_ready(succ);
    }
}

bool Scheduler::is_completed(EventId id) const {
    return m_events.find(id) == m_events.end();
}

bool Scheduler::is_idle() const {
    return m_events.empty();
}

bool Scheduler::is_high_priority(const std::shared_ptr<TaskNode>& node) {
    return std::holds_alternative<CommandBufferCreate>(node->command)
        || std::holds_alternative<CommandBufferDelete>(node->command)
        || std::holds_alternative<CommandEmpty>(node->command);
}

void Scheduler::enqueue_if_ready(const std::shared_ptr<TaskNode>& node) {
    if (node->status != TaskNode::Status::Pending || node->dependencies_not_completed > 0) {
        return;
    }

    node->status = TaskNode::Status::Ready;
    if (is_high_priority(node)) {
        m_ready_high_priority.push_back(node);
    } else {
        m_ready_low_priority.push_back(node);
    }
}

}  // namespace kmm
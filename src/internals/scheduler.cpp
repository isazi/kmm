#include "spdlog/spdlog.h"

#include "kmm/internals/scheduler.hpp"

namespace kmm {

static constexpr size_t NUM_DEFAULT_QUEUES = 3;
static constexpr size_t QUEUE_MISC = 0;
static constexpr size_t QUEUE_BUFFERS = 1;
static constexpr size_t QUEUE_HOST = 2;
static constexpr size_t QUEUE_DEVICES = 3;

struct QueueSlot {
    QueueSlot(std::shared_ptr<TaskNode> node) : node(std::move(node)) {}

    std::shared_ptr<TaskNode> node;
    bool scheduled = false;
    bool completed = false;
    std::shared_ptr<QueueSlot> next = nullptr;
};

struct QueueHeadTail {
    std::shared_ptr<QueueSlot> head;
    std::shared_ptr<QueueSlot> tail;
};

struct Scheduler::Queue {
    std::vector<std::deque<std::shared_ptr<TaskNode>>> jobs;
    size_t max_concurrent_jobs = std::numeric_limits<size_t>::max();
    size_t num_jobs_active;
    std::unordered_map<EventId, QueueHeadTail> node_to_slot;
    std::shared_ptr<QueueSlot> head = nullptr;
    std::shared_ptr<QueueSlot> tail = nullptr;

    void push_job(const TaskNode* predecessor, std::shared_ptr<TaskNode> node);
    bool pop_job(std::shared_ptr<TaskNode>& node);
    void scheduled_job(std::shared_ptr<TaskNode> node);
    void completed_job(std::shared_ptr<TaskNode> node);
};

Scheduler::Scheduler(size_t num_devices) {
    m_queues.resize(NUM_DEFAULT_QUEUES + num_devices);

    for (size_t i = 0; i < num_devices; i++) {
        m_queues[QUEUE_DEVICES + i].max_concurrent_jobs = 1;
    }
}

Scheduler::~Scheduler() = default;

void Scheduler::insert_event(EventId event_id, Command command, const EventList& deps) {
    spdlog::debug("submit event {} (command={}, dependencies={})", event_id, command, deps);

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

    node->queue_id = determine_queue_id(node->command);
    node->dependencies_not_completed = num_not_completed;
    node->dependencies_not_scheduled = num_not_scheduled;
    node->dependency_events = std::move(dependency_events);
    enqueue_if_ready(nullptr, node);

    m_events.emplace(event_id, std::move(node));
}

bool Scheduler::is_completed(EventId id) const {
    return m_events.find(id) == m_events.end();
}

bool Scheduler::is_idle() const {
    return m_events.empty();
}

void Scheduler::set_scheduled(std::shared_ptr<TaskNode> node, CudaEvent event) {
    spdlog::debug(
        "scheduled event {} (command={}, cuda event={})",
        node->id(),
        node->command,
        event);

    KMM_ASSERT(node->status == TaskNode::Status::Scheduled);
    KMM_ASSERT(node->cuda_event == std::nullopt);
    node->cuda_event = event;

    for (const auto& succ : node->successors) {
        succ->dependency_events.push_back(event);
        succ->dependencies_not_scheduled -= 1;
        enqueue_if_ready(node.get(), succ);
    }

    m_queues.at(node->queue_id).scheduled_job(node);
}

void Scheduler::set_complete(std::shared_ptr<TaskNode> node) {
    spdlog::debug("completed event {} (command={})", node->id(), node->command);

    KMM_ASSERT(node->status == TaskNode::Status::Scheduled);
    node->status = TaskNode::Status::Done;
    node->cuda_event = std::nullopt;
    m_events.erase(node->event_id);

    for (const auto& succ : node->successors) {
        succ->dependencies_not_completed -= 1;
        enqueue_if_ready(node.get(), succ);
    }

    m_queues.at(node->queue_id).completed_job(node);
}

std::optional<std::shared_ptr<TaskNode>> Scheduler::pop_ready() {
    std::shared_ptr<TaskNode> result;

    for (auto& q : m_queues) {
        if (q.pop_job(result)) {
            break;
        }
    }

    if (result == nullptr) {
        return std::nullopt;
    }

    spdlog::debug("scheduling event {} (command={})", result->id(), result->command);

    result->status = TaskNode::Status::Scheduled;
    return result;
}

size_t Scheduler::determine_queue_id(const Command& cmd) {
    if (const auto* p = std::get_if<CommandExecute>(&cmd)) {
        if (p->processor_id.is_device()) {
            return QUEUE_DEVICES + p->processor_id.as_device();
        } else {
            return QUEUE_HOST;
        }
    } else if (
        std::holds_alternative<CommandBufferCreate>(cmd)
        || std::holds_alternative<CommandBufferDelete>(cmd)
        || std::holds_alternative<CommandEmpty>(cmd)) {
        return QUEUE_BUFFERS;
    } else {
        return QUEUE_MISC;
    }
}

void Scheduler::enqueue_if_ready(
    const TaskNode* predecessor,
    const std::shared_ptr<TaskNode>& node) {
    if (node->status != TaskNode::Status::Pending || node->dependencies_not_completed > 0) {
        return;
    }

    node->status = TaskNode::Status::Ready;
    m_queues.at(node->queue_id).push_job(predecessor, node);
}

void Scheduler::Queue::push_job(const TaskNode* predecessor, std::shared_ptr<TaskNode> node) {
    auto event_id = node->id();
    auto new_slot = std::make_shared<QueueSlot>(std::move(node));
    node_to_slot.emplace(event_id, QueueHeadTail {new_slot, new_slot});

    if (predecessor != nullptr) {
        auto it = node_to_slot.find(predecessor->id());

        if (it != node_to_slot.end()) {
            KMM_ASSERT(it->second.head->node.get() == predecessor);
            auto current = std::exchange(it->second.tail, new_slot);
            auto next = std::exchange(current->next, new_slot);

            if (next != nullptr) {
                new_slot->next = std::move(next);
            } else {
                tail = new_slot;
            }

            return;
        }
    }

    // Predecessor is not in `node_to_slot`
    auto old_tail = std::exchange(tail, new_slot);

    if (old_tail != nullptr) {
        old_tail->next = new_slot;
    } else {
        head = new_slot;
    }
}

bool Scheduler::Queue::pop_job(std::shared_ptr<TaskNode>& node) {
    if (num_jobs_active >= max_concurrent_jobs) {
        return false;
    }

    auto* p = head.get();

    while (true) {
        if (p == nullptr) {
            return false;
        }

        if (!p->scheduled) {
            break;
        }

        p = p->next.get();
    }

    num_jobs_active++;
    p->scheduled = true;
    node = p->node;
    return true;
}

void Scheduler::Queue::scheduled_job(std::shared_ptr<TaskNode> node) {
    // Nothing to do after scheduling
}

void Scheduler::Queue::completed_job(std::shared_ptr<TaskNode> node) {
    num_jobs_active--;
    auto it = node_to_slot.find(node->id());

    // ???
    if (it == node_to_slot.end()) {
        return;
    }

    auto slot = it->second.head;
    node_to_slot.erase(it);

    // Set slot to completed
    KMM_ASSERT(slot->node == node);
    slot->completed = true;

    // Remove all slots that have been marked as completed
    while (head != nullptr && head->completed) {
        head = head->next;
    }

    if (head == nullptr) {
        tail = nullptr;
    }
}

}  // namespace kmm
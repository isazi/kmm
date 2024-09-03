#pragma once

#include <deque>
#include <unordered_map>

#include "commands.hpp"
#include "cuda_stream_manager.hpp"

namespace kmm {

struct TaskNode {
    friend class Scheduler;

    enum Status { Pending, Ready, Scheduled, Done };

    TaskNode(EventId event_id, Command&& command) :
        event_id(event_id),
        command(std::move(command)) {}

    const Command& get_command() const {
        return command;
    }

  private:
    EventId event_id;
    Status status = Status::Pending;
    Command command;
    std::optional<CudaEvent> cuda_event;
    small_vector<std::shared_ptr<TaskNode>> successors;
    small_vector<CudaEvent> dependency_events;
    size_t dependencies_not_scheduled = 0;
    size_t dependencies_not_completed = 0;
};

class Scheduler {
    KMM_NOT_COPYABLE_OR_MOVABLE(Scheduler)

  public:
    Scheduler() = default;
    void insert_event(EventId event_id, Command command, const EventList& deps = {});
    std::optional<std::shared_ptr<TaskNode>> pop_ready();
    void set_scheduled(std::shared_ptr<TaskNode> node, CudaEvent event);
    void set_complete(std::shared_ptr<TaskNode> node);
    bool is_completed(EventId id) const;
    bool is_idle() const;

  private:
    bool is_high_priority(const std::shared_ptr<TaskNode>& node);
    void enqueue_if_ready(const std::shared_ptr<TaskNode>& node);

    std::deque<std::shared_ptr<TaskNode>> m_ready_high_priority;
    std::deque<std::shared_ptr<TaskNode>> m_ready_low_priority;
    std::unordered_map<EventId, std::shared_ptr<TaskNode>> m_events;
};
}  // namespace kmm
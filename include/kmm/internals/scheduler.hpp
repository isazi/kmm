#pragma once

#include <deque>
#include <unordered_map>

#include "commands.hpp"
#include "device_stream_manager.hpp"

namespace kmm {

class Scheduler {
    KMM_NOT_COPYABLE_OR_MOVABLE(Scheduler)

  public:
    struct Queue;
    struct Node {
        friend class Scheduler;

        enum Status { Pending, Ready, Scheduled, Done };

        Node(EventId event_id, Command&& command) :
            event_id(event_id),
            command(std::move(command)) {}

        EventId id() const {
            return event_id;
        }

        const Command& get_command() const {
            return command;
        }

      private:
        EventId event_id;
        Status status = Status::Pending;
        Command command;
        size_t queue_id;
        std::optional<DeviceEvent> device_event;
        small_vector<std::shared_ptr<Node>, 4> successors;
        DeviceEventSet dependencies_events;
        size_t dependencies_pending = 0;
    };

    Scheduler(size_t num_devices);
    ~Scheduler();

    void insert_event(EventId event_id, Command command, const EventList& deps = {});
    std::optional<std::shared_ptr<Node>> pop_ready(DeviceEventSet* deps_out);

    void set_scheduled(EventId id, DeviceEvent event);
    void set_complete(EventId id);

    bool is_completed(EventId id) const;
    bool is_idle() const;

  private:
    size_t determine_queue_id(const Command& cmd);
    void enqueue_if_ready(const Node* predecessor, const std::shared_ptr<Node>& node);

    std::vector<Queue> m_queues;
    std::unordered_map<EventId, std::shared_ptr<Node>> m_events;
};

using TaskNode = std::shared_ptr<Scheduler::Node>;

}  // namespace kmm
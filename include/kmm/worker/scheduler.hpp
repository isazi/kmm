#pragma once

#include <deque>
#include <map>
#include <memory>
#include <unordered_map>

#include "kmm/event_list.hpp"
#include "kmm/panic.hpp"
#include "kmm/worker/command.hpp"

namespace kmm {

class Scheduler {
    using sequence_number_t = uint64_t;

  public:
    class Node {
        KMM_NOT_COPYABLE_OR_MOVABLE(Node)

      public:
        Node(
            EventId id,
            size_t predecessors,
            Command command,
            std::shared_ptr<Node> parent,
            sequence_number_t sequence_number) :
            identifier(id),
            command(std::move(command)),
            parent(std::move(parent)),
            unsatisfied_predecessors(predecessors),
            sequence_number(sequence_number) {
            if (const auto* cmd = std::get_if<ExecuteCommand>(&this->command)) {
                queue_id = cmd->device_id.get() + 1;
            } else {
                queue_id = 0;
            }
        }

        /**
         * Return the identifier of this task.
         */
        EventId id() const {
            return identifier;
        }

        /**
         * Take the command out of this task.
         */
        Command&& take_command() {
            KMM_ASSERT(status == Status::Running);
            return std::move(command);
        }

      private:
        friend class Scheduler;

        enum class Status { Pending, Ready, Running, WaitingForChildren, Done };
        Status status = Status::Pending;

        EventId identifier;
        Command command;

        std::shared_ptr<Node> parent;
        size_t unsatisfied_predecessors;
        size_t active_children = 0;
        std::vector<std::shared_ptr<Node>> successors = {};

        uint8_t queue_id = 0;
        uint64_t workload = 1;
        sequence_number_t sequence_number;
    };

    /**
     * Construct a new scheduler.
     */
    Scheduler();

    /**
     * Insert a new task into this scheduler.
     *
     * @param id The identifier of the task.
     * @param command The command that the tasks executes.
     * @param dependencies The predecessors that should complete first.
     * @param parent The parent task. Can be null.
     * @return A handle to the new task.
     */
    std::shared_ptr<Node> insert(
        EventId id,
        Command command,
        EventList dependencies,
        std::shared_ptr<Node> parent = nullptr);

    /**
     * Pop a task that is ready to run from the is_ready queue. Returns nullopt if there not task
     * that is is_ready to run.
     */
    std::optional<std::shared_ptr<Node>> pop_ready();

    /**
     * Mark the given task as `completed`. This will trigger the successors of the task to run.
     * @param node The handle returned by `insert`.
     */
    void complete(std::shared_ptr<Node> node);

    /**
     * Checks if the task with the given `id` has completed.
     */
    bool is_completed(EventId id) const;

    /**
     * Checks if all tasks that have been inserted are now completed.
     */
    bool is_all_completed() const;

    /**
     * Returns the identifiers ot the tasks that are currently active.
     *
     * @return
     */
    EventList active_tasks() const;

  private:
    void push_ready(std::shared_ptr<Node> node);
    void satisfy_job_dependencies(std::shared_ptr<Node> node, size_t satisfied = 1);
    void complete_impl(std::shared_ptr<Node> node);

    struct ReadyQueue {
        uint64_t current_workload = 0;
        uint64_t max_workload = 4;  // TODO: why is this hardcoded?
        std::map<sequence_number_t, std::shared_ptr<Node>> ready;
    };

    std::unordered_map<EventId, std::shared_ptr<Node>> m_tasks;
    std::vector<ReadyQueue> m_ready_queues;
    sequence_number_t m_next_sequence_number = 0;
};

}  // namespace kmm
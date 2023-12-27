#pragma once

#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

#include "spdlog/spdlog.h"

#include "kmm/executor.hpp"
#include "kmm/utils.hpp"
#include "kmm/worker/command.hpp"
#include "kmm/worker/memory_manager.hpp"

namespace kmm {

class Scheduler {
  public:
    class Node {
      public:
        Node(EventId id, size_t predecessors, Command command, std::shared_ptr<Node> parent) :
            identifier(id),
            command(std::move(command)),
            parent(std::move(parent)),
            unsatisfied_predecessors(predecessors) {}
        Node(const Node&) = delete;
        Node(Node&&) = delete;

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
    };

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

  private:
    void push_ready(std::shared_ptr<Node> node);
    void satisfy_job_dependencies(std::shared_ptr<Node> node, size_t satisfied = 1);
    void complete_impl(std::shared_ptr<Node> node);

    std::unordered_map<EventId, std::shared_ptr<Node>> m_jobs;
    std::deque<std::shared_ptr<Node>> m_ready_aux;
    std::deque<std::shared_ptr<Node>> m_ready_tasks;
};

}  // namespace kmm
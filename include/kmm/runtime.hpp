#pragma once

#include <chrono>
#include <memory>
#include <utility>

#include "kmm/executor.hpp"
#include "kmm/identifiers.hpp"

namespace kmm {

class RuntimeImpl;

class Runtime {
  public:
    Runtime(std::shared_ptr<RuntimeImpl> impl);

    /**
     * Submit a task to the runtime system.
     *
     * @param task The Task definition.
     * @param reqs The task requirements.
     * @return The event identifier of the submitted task.
     */
    EventId submit_task(std::shared_ptr<Task> task, TaskRequirements reqs) const;

    /**
     * Submit a task to the runtime system using a task launcher.
     *
     * @param launcher The launcher that will submit the task.
     * @param args The arguments that are forwarded to the launcher.
     * @return The event identifier of the submitted task.
     */
    template<typename Launcher, typename... Args>
    EventId submit(const Launcher& launcher, Args&&... args) {
        return launcher(*m_impl, std::forward<Args>(args)...);
    }

    /**
     * Takes a list of events and returns a new event that gets triggered once all the provided
     * events have completed.
     */
    EventId join(EventList events) const;

    /**
     * Takes a list of events and returns a new event that gets triggered once all the provided
     * events have completed.
     *
     * @param events The events. Each argument should be convertible to an EventId.
     */
    template<typename... Es>
    EventId join(Es... events) const {
        return join(EventList {EventId {events}...});
    }

    /**
     * Returns `true` if the event with the provided id has finished, otherwise returns `false`.
     */
    bool is_done(EventId id) const;

    /**
     * Block the current thread until the event with the provided id completes.
     */
    void wait(EventId id) const;

    /**
     * Block the current thread until the event with the provided id completes. Blocks until
     * either the event completes or the deadline is exceeded, whatever comes first.
     *
     * @return `true` if the event with the provided id has finished, otherwise returns `false`.
     */
    bool wait_until(EventId id, typename std::chrono::system_clock::time_point deadline) const;

    /**
     * Block the current thread until the event with the provided id completes. Blocks until
     * either the event completes or the duration is exceeded, whatever comes first.
     *
     * @return `true` if the event with the provided id has finished, otherwise returns `false`.
     */
    bool wait_for(EventId id, typename std::chrono::system_clock::duration duration) const;

    /**
     * Submit a barrier the runtime system. The barrier completes once all the tasks submitted
     * to the runtime system so far have finished.
     *
     * @return The identifier of the barrier.
     */
    EventId submit_barrier() const;

    /**
     * Blocks until all the tasks submitted to the runtime system have finished and the
     * system has become idle.
     */
    void synchronize() const;

    /**
     * Returns the inner `RuntimeImpl`.
     * @return The `RuntimeImpl`.
     */
    std::shared_ptr<RuntimeImpl> inner() const {
        return m_impl;
    }

  private:
    std::shared_ptr<RuntimeImpl> m_impl;
};

Runtime build_runtime();

}  // namespace kmm
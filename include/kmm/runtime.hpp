#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "kmm/executor.hpp"
#include "kmm/panic.hpp"
#include "kmm/types.hpp"

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
     * @param dependencies Events that should complete before the task may run.
     * @return The event identifier of the submitted task.
     */
    EventId submit_task(
        std::shared_ptr<Task> task,
        TaskRequirements reqs,
        EventList dependencies = {}) const;

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

/**
 * Represents a memory buffer within the runtime system. It provides functionalities to prefetch
 * data, synchronize operations, and manage the memory block lifecycle.
 */
class Block {
  public:
    Block(std::shared_ptr<RuntimeImpl> runtime, BlockId id = BlockId::invalid());
    ~Block();
    Block(Block&&) = delete;
    Block(const Block&) = delete;
    Block& operator=(Block&&) = delete;
    Block& operator=(const Block&) = delete;

    /**
     * Returns the unique identifier of this buffer.
     */
    BlockId id() const {
        return m_id;
    }

    /**
     * Returns the runtime associated with this buffer.
     */
    Runtime runtime() const {
        return m_runtime;
    }

    /**
     * Prefetch this buffer in the provided memory.
     *
     * @param memory_id The identifier of the memory.
     * @param dependencies Events that should complete before the prefetch occurs.
     * @return The identifier of the prefetch event.
     */
    EventId prefetch(MemoryId memory_id, EventList dependencies = {}) const;

    /**
     * Submit a barrier the runtime system. The barrier completes once all the events submitted
     * to the runtime system so far associated with this buffer have finished execution.
     *
     * @return The identifier of the barrier.
     */
    EventId submit_barrier() const;

    /**
     * Blocks until all the events associated with this buffer have finished execution.
     */
    void synchronize() const;

    /**
     * Delete this buffer. It is not necessary to call this method manually, since it will also be
     * called by the destructor.
     */
    void destroy();

    /**
     *
     */
    BlockId release();

  private:
    BlockId m_id = BlockId::invalid();
    std::shared_ptr<RuntimeImpl> m_runtime;
};

Runtime build_runtime();

}  // namespace kmm
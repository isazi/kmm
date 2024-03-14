#pragma once

#include <chrono>
#include <memory>
#include <utility>

#include "kmm/array.hpp"
#include "kmm/device.hpp"
#include "kmm/identifiers.hpp"
#include "kmm/task_argument.hpp"
#include "kmm/utils/checked_math.hpp"
#include "kmm/utils/view.hpp"

namespace kmm {

class Runtime;

class RuntimeHandle {
  public:
    RuntimeHandle(std::shared_ptr<Runtime> impl);

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
    EventId submit(Launcher launcher, Args&&... args) const {
        DeviceId device_id = launcher.find_device(*m_impl);

        return submit_task_with_launcher(
            *m_impl,
            device_id,
            std::move(launcher),
            std::forward<Args>(args)...);
    }

    /**
     * Find the device identifier that would be used by the provided task launcher.
     */
    template<typename Launcher>
    DeviceId find_device(const Launcher& launcher) const {
        return launcher.find_device(*m_impl);
    }

    /**
     * Returns the memory identifier that has the best affinity for the given memory address.
     */
    MemoryId memory_affinity_for_address(const void* address) const;

    /**
     * Create a new array where the data is provided by memory stored on the host. The dimensions
     * of the new array will be `sizes`. The provided buffer must contain exactly
     * `sizes[0] * sizes[1] * ...` elements.
     *
     * @param data_ptr The provided data elements.
     * @param sizes The dimensions of the new array.
     * @return The new array.
     */
    template<typename T, size_t N>
    Array<T, N> allocate_array(const T* data_ptr, std::array<index_t, N> sizes) const {
        index_t num_elements = checked_product(sizes.begin(), sizes.end());
        size_t num_bytes = checked_mul(checked_cast<size_t>(num_elements), sizeof(T));

        auto memory_id = memory_affinity_for_address(data_ptr);
        auto header = std::make_unique<ArrayHeader>(ArrayHeader::for_type<T>(num_elements));

        auto block_id = m_impl->create_block(memory_id, std::move(header), data_ptr, num_bytes);
        auto block = std::make_shared<Block>(m_impl, block_id);

        return {sizes, block};
    }

    /**
     * Create a new array where the data is provided by memory stored on the host. The dimensions
     * of the new array will be `{sizes[0], sizes[1], ...}`. The provided buffer must contain
     * exactly `sizes[0] * sizes[1] * ...` elements.
     *
     * @param data_ptr The provided data elements.
     * @param sizes The dimensions of the new array.
     * @return The new array.
     */
    template<typename T, typename... Sizes, size_t N = sizeof...(Sizes)>
    Array<T, N> allocate(const T* data_ptr, Sizes... sizes) const {
        return allocate_array(data_ptr, std::array<index_t, N> {checked_cast<index_t>(sizes)...});
    }

    /**
     * This is an alias for `allocate(vector.data(), vector.size())`
     */
    template<typename T>
    Array<T> allocate(const std::vector<T>& vector) const {
        return allocate(vector.data(), vector.size());
    }

    /**
     * This is an alias for `allocate(input.data(), input.sizes())`
     */
    template<typename T, size_t N>
    Array<T> allocate(view<T, N> input) const {
        std::array<index_t, N> shape;
        for (size_t i = 0; i < N; i++) {
            shape[i] = input.size(i);
        }

        return allocate_array(input.data(), shape);
    }

    /**
     * This is an alias for `allocate(list.data(), list.size())`
     */
    template<typename T>
    Array<T> allocate(std::initializer_list<T> list) const {
        return allocate(list.begin(), list.size());
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
    std::shared_ptr<Runtime> inner() const {
        return m_impl;
    }

  private:
    std::shared_ptr<Runtime> m_impl;
};
/**
 *
 * Initialize the KMM runtime.
 *
 * @return A handler to interact with the KMM runtime.
 */
RuntimeHandle build_runtime();

}  // namespace kmm
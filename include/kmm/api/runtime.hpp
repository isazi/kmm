#pragma once

#include <array>
#include <chrono>
#include <memory>
#include <vector>

#include "array.hpp"
#include "launcher.hpp"
#include "partition.hpp"
#include "runtime_impl.hpp"

#include "kmm/core/device_info.hpp"
#include "kmm/core/identifiers.hpp"
#include "kmm/utils/panic.hpp"

namespace kmm {

class Runtime {
  public:
    Runtime(std::shared_ptr<RuntimeImpl> impl) : m_impl(impl) {}

    template<typename Partitioner, typename Launcher, typename... Args>
    EventId parallel_submit(Partitioner partition, Launcher launcher, Args&&... args) {
        return submit_with_launcher(m_impl.get(), partition(m_impl.get()), launcher, args...);
    }

    template<typename Launcher, typename... Args>
    EventId submit(DeviceId device_id, Launcher launcher, Args&&... args) {
        auto partition = Partition<0> {dim<0> {}, {Chunk<0> {rect<0> {}, device_id}}};
        return submit_with_launcher(m_impl.get(), partition, launcher, args...);
    }

    template<typename T, size_t N>
    Array<T, N> allocate_array(const T* data_ptr, std::array<index_t, N> sizes) const {
        auto num_elements = checked_product(sizes.begin(), sizes.end());
        BufferLayout layout = BufferLayout::for_type<T>(num_elements);

        KMM_TODO();
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
    //    template<typename T, size_t N>
    //    Array<T> allocate(view<T, N> input) const {
    //        std::array<index_t, N> shape;
    //        for (size_t i = 0; i < N; i++) {
    //            shape[i] = input.size(i);
    //        }
    //
    //        return allocate_array(input.data(), shape);
    //    }

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
     * Returns information on the available devices.
     */
    const std::vector<DeviceInfo>& devices() const;

    /**
     * Returns the number of available devices.
     */
    size_t num_devices() const {
        return devices().size();
    }

    /**
     * Returns information about the device `id`.
     */
    const DeviceInfo& device(DeviceId id) const {
        return devices().at(id);
    }

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

Runtime make_runtime();

}  // namespace kmm
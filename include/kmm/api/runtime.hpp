#pragma once

#include <memory>
#include <vector>

#include "array.hpp"
#include "launcher.hpp"

#include "kmm/core/system_info.hpp"
#include "kmm/core/view.hpp"
#include "kmm/core/work_range.hpp"
#include "kmm/utils/checked_math.hpp"
#include "kmm/utils/panic.hpp"

namespace kmm {

class Worker;

class Runtime {
  public:
    Runtime(std::shared_ptr<Worker> worker);
    Runtime(Worker& worker);

    /**
     * Submit a single task to the runtime system.
     *
     * @param index_space The index space defining the task dimensions.
     * @param target The target processor for the task.
     * @param launcher The task launcher.
     * @param args The arguments that are forwarded to the launcher.
     * @return The event identifier for the submitted task.
     */
    template<typename L, typename... Args>
    EventId submit(WorkDim index_space, ProcessorId target, L launcher, Args&&... args) {
        TaskPartition partition = {{TaskChunk {
            .owner_id = target,  //
            .offset = {},
            .size = index_space}}};

        return kmm::parallel_submit(
            m_worker,
            info(),
            partition,
            launcher,
            std::forward<Args>(args)...
        );
    }

    /**
     * Submit a set of tasks to the runtime systems.
     *
     * @param partition The partition describing how the work is split.
     * @param launcher The task launcher.
     * @param args The arguments that are forwarded to the launcher.
     * @return The event identifier for the submitted task.
     */
    template<typename L, typename... Args>
    EventId parallel_submit(TaskPartition partition, L launcher, Args&&... args) {
        return kmm::parallel_submit(
            m_worker,
            info(),
            partition,
            launcher,
            std::forward<Args>(args)...
        );
    }

    /**
     * Submit a set of tasks to the runtime systems.
     *
     * @param index_space The index space defining the loop dimensions.
     * @param partitioner The partitioner describing how the work is split.
     * @param launcher The task launcher.
     * @param args The arguments that are forwarded to the launcher.
     * @return The event identifier for the submitted task.
     */
    template<typename P = TaskPartitioner, typename L, typename... Args>
    EventId parallel_submit(WorkDim index_space, P partitioner, L launcher, Args&&... args) {
        return kmm::parallel_submit(
            m_worker,
            info(),
            partitioner(index_space, info()),
            launcher,
            std::forward<Args>(args)...
        );
    }

    /**
     * Allocates an array in memory with the given shape and memory affinity.
     *
     * The pointer to the given buffer should contain `shape[0] * shape[1] * shape[2]...`
     * elements.
     *
     * @param data Pointer to the array data.
     * @param shape Shape of the array.
     * @param memory_id Identifier of the memory region.
     * @return The allocated Array object.
     */
    template<size_t N = 1, typename T>
    Array<T, N> allocate(const T* data, Dim<N> shape, MemoryId memory_id) {
        BufferLayout layout = BufferLayout::for_type<T>(checked_cast<size_t>(shape.volume()));
        BufferId buffer_id = allocate_bytes(data, layout, memory_id);

        std::vector<ArrayChunk<N>> chunks = {{buffer_id, memory_id, Point<N>::zero(), shape}};
        return std::make_shared<ArrayBackend<N>>(m_worker, shape, chunks);
    }

    /**
     * Allocates an array in memory with the given shape.
     *
     * The pointer to the given buffer should contain `shape[0] * shape[1] * shape[2]...`
     * elements.
     *
     * In which memory the data will be allocated is determined by `memory_affinity_for_address`.
     *
     * @param data Pointer to the array data.
     * @param shape Shape of the array.
     * @return The allocated Array object.
     */
    template<size_t N = 1, typename T>
    Array<T, N> allocate(const T* data, Dim<N> shape) {
        return allocate(data, shape, memory_affinity_for_address(data));
    }

    /**
     * Alias for `allocate(v.data(), v.sizes())`
     */
    template<typename T, size_t N = 1>
    Array<T, N> allocate(view<T, N> v) {
        return allocate(v.data(), v.sizes());
    }

    /**
     * Alias for `allocate(data, dim<N>{sizes...})`
     */
    template<typename T, typename... Sizes>
    Array<T> allocate(const T* data, Sizes... num_elements) {
        return allocate(data, Dim<sizeof...(Sizes)> {checked_cast<int64_t>(num_elements)...});
    }

    /**
     * Alias for `allocate(v.data(), v.size())`
     */
    template<typename T>
    Array<T> allocate(const std::vector<T>& v) {
        return allocate(v.data(), v.size());
    }

    /**
     * Alias for `allocate(v.begin(), v.size())`
     */
    template<typename T>
    Array<T> allocate(std::initializer_list<T> v) {
        return allocate(v.begin(), v.size());
    }

    /**
     * Returns the memory affinity for a given address.
     */
    MemoryId memory_affinity_for_address(const void* address) const;

    BufferId allocate_bytes(const void* data, BufferLayout layout, MemoryId memory_id);

    /**
     * Returns `true` if the event with the provided identifier has finished, or `false` otherwise.
     */
    bool is_done(EventId) const;

    /**
     * Block the current thread until the event with the provided identifier completes.
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
    EventId barrier() const;

    /**
     * Blocks until all the tasks submitted to the runtime system have finished and the
     * system has become idle.
     */
    void synchronize() const;

    /**
     * Returns information about the current system.
     */
    const SystemInfo& info() const;

    /**
     * Returns the inner `Worker`.
     */
    const Worker& worker() const;

  private:
    std::shared_ptr<Worker> m_worker;
};

Runtime make_runtime(const WorkerConfig& config = default_config_from_environment());

}  // namespace kmm
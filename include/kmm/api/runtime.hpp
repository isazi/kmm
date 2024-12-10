#pragma once

#include <memory>
#include <vector>

#include "array.hpp"
#include "launcher.hpp"

#include "kmm/core/domain.hpp"
#include "kmm/core/system_info.hpp"
#include "kmm/core/view.hpp"
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
    EventId submit(NDRange index_space, ProcessorId target, L&& launcher, Args&&... args) const {
        TaskChunk chunk = {
            .owner_id = target,  //
            .offset = index_space.begin,
            .size = index_space.sizes()};

        return kmm::parallel_submit(
            m_worker,
            info(),
            TaskPartition {{chunk}},
            std::forward<L>(launcher),
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
    EventId parallel_submit(TaskPartition partition, L&& launcher, Args&&... args) const {
        return kmm::parallel_submit(
            m_worker,
            info(),
            std::move(partition),
            std::forward<L>(launcher),
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
    EventId parallel_submit(NDRange index_space, P&& partitioner, L&& launcher, Args&&... args)
        const {
        return kmm::parallel_submit(
            m_worker,
            info(),
            partitioner(index_space, info(), std::decay_t<L>::execution_space),
            std::forward<L>(launcher),
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
    Array<T, N> allocate(const T* data, Size<N> shape, MemoryId memory_id) const {
        auto layout = BufferLayout::for_type<T>().repeat(checked_cast<size_t>(shape.volume()));
        auto buffer_id = allocate_bytes(data, layout, memory_id);

        auto chunk = DataChunk<N> {buffer_id, memory_id, Index<N>::zero(), shape};
        auto backend =
            std::make_shared<ArrayHandle<N>>(m_worker, shape, std::vector<DataChunk<N>> {chunk});
        return Array<T, N> {std::move(backend)};
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
    Array<T, N> allocate(const T* data, const Size<N>& shape) const {
        return allocate(data, shape, memory_affinity_for_address(data));
    }

    /**
     * Alias for `allocate(v.data(), v.sizes())`
     */
    template<typename T, size_t N = 1>
    Array<T, N> allocate(view<T, N> v) const {
        return allocate(v.data(), v.sizes());
    }

    /**
     * Alias for `allocate(data, dim<N>{sizes...})`
     */
    template<typename T, typename... Is>
    Array<T, sizeof...(Is)> allocate(const T* data, const Is&... num_elements) const {
        return allocate(data, Size<sizeof...(Is)> {checked_cast<int64_t>(num_elements)...});
    }

    /**
     * Alias for `allocate(v.data(), v.size())`
     */
    template<typename T>
    Array<T> allocate(const std::vector<T>& v) const {
        return allocate(v.data(), v.size());
    }

    /**
     * Alias for `allocate(v.begin(), v.size())`
     */
    template<typename T>
    Array<T> allocate(std::initializer_list<T> v) const {
        return allocate(v.begin(), v.size());
    }

    /**
     * Returns the memory affinity for a given address.
     */
    MemoryId memory_affinity_for_address(const void* address) const;

    BufferId allocate_bytes(const void* data, BufferLayout layout, MemoryId memory_id) const;

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

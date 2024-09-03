#pragma once

#include <memory>
#include <vector>

#include "array.hpp"
#include "launcher.hpp"

#include "kmm/core/system_info.hpp"
#include "kmm/core/view.hpp"
#include "kmm/internals/worker.hpp"
#include "kmm/utils/checked_math.hpp"
#include "kmm/utils/panic.hpp"

namespace kmm {

class Worker;

class Runtime {
  public:
    Runtime(std::shared_ptr<Worker> worker) : m_worker(std::move(worker)) {
        KMM_ASSERT(worker != nullptr);
    }

    Runtime(Worker& worker) : Runtime(worker.shared_from_this()) {}

    template<typename P, typename L, typename... Args>
    EventId parallel_submit(P partition, L launcher, Args&&... args) {
        return kmm::parallel_submit(
            *m_worker,
            partition(info()),
            launcher,
            std::forward<Args>(args)...);
    }

    MemoryId memory_affinity_for_address(const void* address) const;

    template<size_t N = 1, typename T>
    Array<T, N> allocate(const T* data, dim<N> shape) {
        MemoryId memory_id = memory_affinity_for_address(data);
        BufferLayout layout = BufferLayout::for_type<T>(checked_cast<size_t>(shape.volume()));
        BufferId buffer_id = allocate_bytes(data, layout, memory_id);

        std::vector<ArrayChunk<N>> chunks = {{buffer_id, memory_id, point<N>::zero(), shape}};

        return std::make_shared<ArrayBackend<N>>(m_worker, shape, chunks);
    }

    template<typename T, size_t N = 1>
    Array<T, N> allocate(view<T, N> data) {
        return allocate(data.data(), data.sizes());
    }

    template<typename T, typename... Sizes>
    Array<T> allocate(const T* data, Sizes... num_elements) {
        return allocate(data, dim<sizeof...(Sizes)> {checked_cast<int64_t>(num_elements)...});
    }

    template<typename T>
    Array<T> allocate(const std::vector<T>& v) {
        return allocate(v.data(), v.size());
    }

    template<typename T>
    Array<T> allocate(std::initializer_list<T> v) {
        return allocate(v.begin(), v.size());
    }

    BufferId allocate_bytes(const void* data, BufferLayout layout, MemoryId memory_id);

    /**
     * Returns `true` if the event with the provided id has finished, otherwise returns `false`.
     */
    bool is_done(EventId) const;

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

Runtime make_runtime();

}  // namespace kmm
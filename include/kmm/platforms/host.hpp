#pragma once

#include <thread>

#include "kmm/executor.hpp"
#include "kmm/memory.hpp"
#include "kmm/runtime.hpp"
#include "kmm/runtime_impl.hpp"
#include "kmm/task_args.hpp"
#include "kmm/types.hpp"

namespace kmm {

class ParallelExecutorContext: public ExecutorContext {};

class ParallelExecutor: public Executor {
  public:
    struct Job;

    ParallelExecutor();
    ~ParallelExecutor() override;
    void submit(std::shared_ptr<Task>, TaskContext, TaskCompletion) override;
    void copy_async(
        const void* src_ptr,
        void* dst_ptr,
        size_t nbytes,
        std::unique_ptr<MemoryCompletion> completion) const;

  private:
    struct Queue;
    std::shared_ptr<Queue> m_queue;
    std::thread m_thread;
};

class HostAllocation: public MemoryAllocation {
  public:
    explicit HostAllocation(size_t nbytes);

    void* data() const {
        return m_data.get();
    }

    size_t size() const {
        return m_nbytes;
    }

  private:
    size_t m_nbytes;
    std::unique_ptr<char[]> m_data;
};

class HostMemory: public Memory {
  public:
    HostMemory(
        std::shared_ptr<ParallelExecutor> executor,
        size_t max_bytes = std::numeric_limits<size_t>::max());

    std::optional<std::unique_ptr<MemoryAllocation>> allocate(DeviceId id, size_t num_bytes)
        override;

    void deallocate(DeviceId id, std::unique_ptr<MemoryAllocation> allocation) override;

    bool is_copy_possible(DeviceId src_id, DeviceId dst_id) override;

    void copy_async(
        DeviceId src_id,
        const MemoryAllocation* src_alloc,
        size_t src_offset,
        DeviceId dst_id,
        const MemoryAllocation* dst_alloc,
        size_t dst_offset,
        size_t num_bytes,
        std::unique_ptr<MemoryCompletion> completion) override;

  private:
    std::shared_ptr<ParallelExecutor> m_executor;
    size_t m_bytes_remaining;
};

struct Host {
    template<typename Fun, typename... Args>
    OperationId operator()(RuntimeImpl& rt, Fun&& fun, Args&&... args) const {
        auto device_id = DeviceId(0);
        auto reqs = TaskRequirements(device_id);

        std::shared_ptr<Task> task = std::make_shared<TaskImpl<
            ExecutionSpace::Host,
            std::decay_t<Fun>,
            typename TaskArgPack<ExecutionSpace::Host, std::decay_t<Args>>::type...>>(
            std::forward<Fun>(fun),
            TaskArgPack<ExecutionSpace::Host, std::decay_t<Args>>::call(
                std::forward<Args>(args),
                reqs)...);

        return rt.submit_task(std::move(task), std::move(reqs));
    }
};

template<typename T, size_t N>
struct PackedArray {
    size_t buffer_index;
};

template<typename T, size_t N>
struct TaskArgPack<ExecutionSpace::Host, Array<T, N>> {
    using type = PackedArray<const T, N>;

    static type call(const Array<T>& array, TaskRequirements& requirements) {
        size_t index = requirements.buffers.size();
        requirements.buffers.push_back(VirtualBufferRequirement {
            .buffer_id = array.id(),
            .mode = AccessMode::Read,
        });

        return {index};
    }
};

template<typename T, size_t N>
struct TaskArgPack<ExecutionSpace::Host, Write<Array<T, N>>> {
    using type = PackedArray<T, N>;

    static type call(const Write<Array<T, N>>& array, TaskRequirements& requirements) {
        size_t index = requirements.buffers.size();
        requirements.buffers.push_back(VirtualBufferRequirement {
            .buffer_id = array.inner.id(),
            .mode = AccessMode::Write,
        });

        return {index};
    }
};

template<typename T, size_t N>
struct TaskArgUnpack<ExecutionSpace::Host, PackedArray<T, N>> {
    static T* call(const PackedArray<T, N>& array, TaskContext& context) {
        auto access = context.buffers.at(array.buffer_index);
        auto& alloc = dynamic_cast<const HostAllocation&>(*access.allocation);
        return reinterpret_cast<T*>(alloc.data());
    }
};

}  // namespace kmm
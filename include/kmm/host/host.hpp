#pragma once

#include <condition_variable>
#include <mutex>
#include <thread>

#include "kmm/executor.hpp"
#include "kmm/memory.hpp"
#include "kmm/runtime.hpp"
#include "kmm/runtime_impl.hpp"
#include "kmm/task_serialize.hpp"
#include "kmm/types.hpp"
#include "kmm/work_queue.hpp"

namespace kmm {

class ParallelExecutorContext: public ExecutorContext {};

class ParallelExecutor: public Executor {
  public:
    ParallelExecutor();
    ~ParallelExecutor() override;
    void submit(std::shared_ptr<Task>, TaskContext, Completion) override;
    void submit_job(std::unique_ptr<WorkQueue<ParallelExecutorContext>::Job> job);

  private:
    std::shared_ptr<WorkQueue<ParallelExecutorContext>> m_queue;
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

    std::optional<std::unique_ptr<MemoryAllocation>> allocate(MemoryId id, size_t num_bytes)
        override;

    void deallocate(MemoryId id, std::unique_ptr<MemoryAllocation> allocation) override;

    bool is_copy_possible(MemoryId src_id, MemoryId dst_id) override;

    void copy_async(
        MemoryId src_id,
        const MemoryAllocation* src_alloc,
        size_t src_offset,
        MemoryId dst_id,
        const MemoryAllocation* dst_alloc,
        size_t dst_offset,
        size_t num_bytes,
        Completion completion) override;

    void fill_async(
        MemoryId dst_id,
        const MemoryAllocation* dst_alloc,
        size_t dst_offset,
        size_t num_bytes,
        std::vector<uint8_t> fill_bytes,
        Completion completion) override;

  private:
    std::shared_ptr<ParallelExecutor> m_executor;
    size_t m_bytes_remaining;
};

struct Host {
    template<typename Fun, typename... Args>
    EventId operator()(RuntimeImpl& rt, Fun&& fun, Args&&... args) const {
        return TaskLauncher<ExecutionSpace::Host, Fun, Args...>::call(
            ExecutorId(0),
            rt,
            fun,
            args...);
    }
};

}  // namespace kmm
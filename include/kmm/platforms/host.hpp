#pragma once

#include <thread>

#include "kmm/executor.hpp"
#include "kmm/memory.hpp"

namespace kmm {

class ParallelExecutorContext: public ExecutorContext {};

class ParallelExecutor: public Executor {
  public:
    class Job {
      public:
        virtual ~Job() = default;
        virtual void execute() = 0;
        std::unique_ptr<Job> next;
    };

    ParallelExecutor();
    ~ParallelExecutor() override;
    void submit(std::shared_ptr<Task>, TaskContext, TaskCompletion) override;
    void copy_async(const void* src_ptr, void* dst_ptr, size_t nbytes, std::unique_ptr<Completion> completion) const;

  private:
    struct Queue;
    std::shared_ptr<Queue> m_queue;
    std::thread m_thread;
};

class HostAllocation: public Allocation {
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
    HostMemory(std::shared_ptr<ParallelExecutor> executor, size_t max_bytes = std::numeric_limits<size_t>::max());

    std::optional<std::unique_ptr<Allocation>> allocate(
        DeviceId id,
        size_t num_bytes) override;

    void deallocate(DeviceId id, std::unique_ptr<Allocation> allocation) override;

    bool is_copy_possible(DeviceId src_id, DeviceId dst_id) override;

    void copy_async(
        DeviceId src_id,
        const Allocation* src_alloc,
        size_t src_offset,
        DeviceId dst_id,
        const Allocation* dst_alloc,
        size_t dst_offset,
        size_t num_bytes,
        std::unique_ptr<Completion> completion) override;

  private:
    std::shared_ptr<ParallelExecutor> m_executor;
    size_t m_bytes_remaining;
};


}
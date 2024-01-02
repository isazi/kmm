#pragma once

#include "kmm/executor.hpp"
#include "kmm/memory.hpp"
#include "kmm/types.hpp"

namespace kmm {

class CudaExecutorContext: public ExecutorContext {};

class CudaExecutor: public Executor {
  public:
    CudaExecutor();
    ~CudaExecutor() override;
    void submit(std::shared_ptr<Task>, TaskContext, TaskCompletion) override;
    void copy_async(
        const void* src_ptr,
        void* dst_ptr,
        size_t nbytes,
        std::unique_ptr<MemoryCompletion> completion) const;

  private:
    std::shared_ptr<WorkQueue<CudaExecutorContext>> m_queue;
};

class CudaAllocation: public MemoryAllocation {
  public:
    explicit CudaAllocation(size_t nbytes);
    ~CudaAllocation();

    void* data() const {
        return m_data;
    }

    size_t size() const {
        return m_nbytes;
    }

  private:
    size_t m_nbytes;
    void* m_data;
};

class CudaMemory: public Memory {
  public:
    std::optional<std::unique_ptr<MemoryAllocation>> allocate(MemoryId id, size_t nbytes) override;

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
        MemoryCompletion completion) override;
};

}  // namespace kmm

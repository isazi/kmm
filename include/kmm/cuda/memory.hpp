#pragma once

#include <condition_variable>
#include <cuda.h>
#include <memory>
#include <mutex>

#include "kmm/completion.hpp"
#include "kmm/cuda/types.hpp"
#include "kmm/executor.hpp"
#include "kmm/host/memory.hpp"
#include "kmm/host/thread_pool.hpp"

namespace kmm {

// Forward so we don't need to include cuda/copy_engine.hpp
class CudaCopyEngine;

class CudaAllocation final: public MemoryAllocation {
  public:
    void* data() const {
        KMM_TODO();
    }
    size_t size() const {
        KMM_TODO();
    }
};

class PinnedAllocation: public HostAllocation {};

class CudaMemory final: public Memory {
  public:
    CudaMemory(std::shared_ptr<ThreadPool> host_thread, std::vector<CudaContextHandle> contexts);
    ~CudaMemory();

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
        std::vector<uint8_t> fill_pattern,
        Completion completion) override;

  private:
    static void* unpack_allocation(
        const MemoryAllocation* alloc,
        size_t offset,
        size_t num_bytes,
        MemoryId memory_id);

    std::shared_ptr<ThreadPool> m_pool;
    std::shared_ptr<CudaCopyEngine> m_copy_engine;
    std::thread m_copy_thread;
};

}  // namespace kmm
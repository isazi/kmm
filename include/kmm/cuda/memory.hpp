#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>

#ifdef KMM_USE_CUDA
    #include <cuda.h>
#endif

#include "kmm/cuda/allocator.hpp"
#include "kmm/cuda/types.hpp"
#include "kmm/device.hpp"
#include "kmm/host/memory.hpp"
#include "kmm/host/thread_pool.hpp"
#include "kmm/utils/completion.hpp"

#ifdef KMM_USE_CUDA

namespace kmm {

// Forward so we don't need to include cuda/copy_engine.hpp
class CudaCopyEngine;

class CudaAllocation final: public MemoryAllocation {
  public:
    CudaAllocation(void* data, size_t size) : m_data(data), m_size(size) {}

    void copy_from_host_sync(const void* src_addr, size_t dst_offset, size_t num_bytes) override;

    void copy_to_host_sync(size_t src_offset, void* dst_addr, size_t num_bytes) const override;

    void* data() const {
        return m_data;
    }

    size_t size() const {
        return m_size;
    }

  private:
    void* m_data;
    size_t m_size;
};

class PinnedAllocation: public HostAllocation {
  public:
    PinnedAllocation(void* data, size_t size) : m_data(data), m_size(size) {}

    void* data() const final {
        return m_data;
    }

    size_t size() const final {
        return m_size;
    }

  private:
    void* m_data;
    size_t m_size;
};

class CudaMemory final: public Memory {
  public:
    CudaMemory(std::shared_ptr<ThreadPool> host_thread, std::vector<CudaContextHandle> contexts);
    ~CudaMemory() override;

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

    std::unique_ptr<CudaPinnedAllocator> m_host_allocator;
    std::vector<std::unique_ptr<CudaDeviceAllocator>> m_device_allocators;
};

}  // namespace kmm

#endif  // KMM_USE_CUDA
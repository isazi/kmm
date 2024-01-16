#pragma once

#include "kmm/host/thread_pool.hpp"
#include "kmm/memory.hpp"

namespace kmm {

class HostAllocation: public MemoryAllocation {
  public:
    virtual void* data() const = 0;
    virtual size_t size() const = 0;

    void copy_from_host_sync(const void* src_addr, size_t dst_offset, size_t num_bytes) override;

    void copy_to_host_sync(size_t src_offset, void* dst_addr, size_t num_bytes) const override;
};

class HostMemory final: public Memory {
  public:
    HostMemory(
        std::shared_ptr<ThreadPool> pool,
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
        std::vector<uint8_t> fill_pattern,
        Completion completion) override;

  private:
    std::shared_ptr<ThreadPool> m_pool;
    size_t m_bytes_remaining;
};

}  // namespace kmm
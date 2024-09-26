#pragma once

#include "base.hpp"
#include <unordered_map>
#include <memory>

namespace kmm {

class CachingMemoryAllocator {
  public:
    CachingMemoryAllocator(std::unique_ptr<MemoryAllocator> allocator);
    ~CachingMemoryAllocator();
    bool allocate(size_t nbytes, void*& addr_out, GPUEventSet& deps_out);
    void deallocate(void* addr, size_t nbytes, GPUEventSet deps);
    size_t free_some_memory();

  private:
    struct AllocationSlot;

    std::unique_ptr<MemoryAllocator> m_allocator;
    std::unordered_map<size_t, std::pair<std::unique_ptr<AllocationSlot>, AllocationSlot*>> m_allocations;
    AllocationSlot* m_lru_oldest = nullptr;
    AllocationSlot* m_lru_newest = nullptr;
    size_t m_bytes_watermark = 0;
    size_t m_bytes_in_use = 0;
    size_t m_bytes_allocated = 0;
};

}
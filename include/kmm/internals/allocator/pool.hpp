#pragma once

#include "base.hpp"
#include <set>
#include <unordered_map>

namespace kmm {


class PoolAllocator: public MemoryAllocator {
  public:
    static constexpr size_t DEFAULT_BLOCK_SIZE = 1024 * 1024 * 500;

    PoolAllocator(std::unique_ptr<MemoryAllocator> allocator, size_t min_block_size=DEFAULT_BLOCK_SIZE);
    ~PoolAllocator();
    bool allocate(size_t nbytes, void*& addr_out, GPUEventSet& deps_out) final;
    void deallocate(void* addr, size_t nbytes, GPUEventSet deps) final;

  private:
    struct Block;
    struct Region;
    struct RegionSize {
        size_t value;
    };

    struct RegionSizeCompare {
        using is_transparent = void;
        bool operator()(Region*, Region*) const;
        bool operator()(Region*, RegionSize) const;
    };

    std::pair<Region*, Region*> split_region(Region* region, size_t left_size);

    std::unique_ptr<MemoryAllocator> m_allocator;
    std::set<Region*, RegionSizeCompare> m_free_regions;
    std::unordered_map<void*, Region*> m_used_regions;
    std::vector<std::unique_ptr<Block>> m_blocks;
    size_t m_min_block_size;
};
}
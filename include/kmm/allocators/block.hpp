#pragma once

#include <set>
#include <unordered_map>

#include "base.hpp"

namespace kmm {

class BlockAllocator: public AsyncAllocator {
  public:
    static constexpr size_t DEFAULT_BLOCK_SIZE = 1024L * 1024 * 500;

    BlockAllocator(
        std::unique_ptr<AsyncAllocator> allocator,
        size_t min_block_size = DEFAULT_BLOCK_SIZE
    );
    ~BlockAllocator();
    bool allocate_async(size_t nbytes, void** addr_out, DeviceEventSet* deps_out) final;
    void deallocate_async(void* addr, size_t nbytes, DeviceEventSet deps) final;
    void make_progress() final;
    void trim(size_t nbytes_remaining = 0) final;

  private:
    struct Block;
    struct Region;
    struct RegionSize {
        size_t value;
    };

    struct RegionSizeCompare {
        using is_transparent = void;
        bool operator()(const Region*, const Region*) const;
        bool operator()(const Region*, RegionSize) const;
    };

    static std::pair<Region*, Region*> split_region(Region* region, size_t left_size);
    static Region* merge_regions(Region* left, Region* right);

    std::unique_ptr<AsyncAllocator> m_allocator;
    std::unordered_map<void*, Region*> m_active_regions;
    std::vector<std::unique_ptr<Block>> m_blocks;
    size_t m_active_block = 0;
    size_t m_min_block_size;
};
}  // namespace kmm
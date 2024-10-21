#include "kmm/internals/allocator/pool.hpp"
#include "kmm/utils/integer_fun.hpp"

namespace kmm {


PoolAllocator::PoolAllocator(std::unique_ptr<MemoryAllocator> allocator, size_t min_block_size):
    m_allocator(std::move(allocator)), m_min_block_size(min_block_size) {
}

PoolAllocator::~PoolAllocator() {
    KMM_TODO();
}

struct PoolAllocator::Region {
    Block* parent = nullptr;
    std::unique_ptr<Region> next = nullptr;
    Region* prev = nullptr;
    size_t offset_in_block = 0;
    size_t size = 0;
    bool in_use = true;
    DeviceEventSet dependencies;

    Region(Block* parent, size_t offset_in_block, size_t size, DeviceEventSet deps={}) {
        this->parent = parent;
        this->offset_in_block = offset_in_block;
        this->size = size;
        this->dependencies = std::move(deps);
    }
};

bool PoolAllocator::RegionSizeCompare::operator()(Region* a, Region* b) const {
    return a->size < b->size;
}

bool PoolAllocator::RegionSizeCompare::operator()(Region* a, RegionSize b) const {
    return a->size < b.value;
}

struct PoolAllocator::Block {
    std::unique_ptr<Region> head = nullptr;
    Region* tail = nullptr;
    void* base_addr = nullptr;
    size_t size = 0;

    Block(void* addr, size_t size, DeviceEventSet deps) {
        this->head = std::make_unique<Region>(this, 0, size, std::move(deps));
        this->tail = head.get();
        this->base_addr = addr;
        this->size = size;
    }
};

static constexpr size_t MAX_ALIGNMENT = 256;

bool PoolAllocator::allocate(size_t nbytes, void*& addr_out, DeviceEventSet& deps_out) {
    size_t alignment = nbytes < MAX_ALIGNMENT ? round_up_to_power_of_two(nbytes) : MAX_ALIGNMENT;
    nbytes = round_up_to_multiple(nbytes, alignment);

    auto it = m_free_regions.lower_bound(RegionSize{ nbytes });

    while (it != m_free_regions.end()) {
        auto& region = **it;
        auto padding = round_up_to_multiple(region.offset_in_block, alignment) - region.offset_in_block;

        if (region.size >= padding + nbytes) {
            break;
        }
    }

    Region *region = nullptr;

    if (it == m_free_regions.end()) {
        DeviceEventSet deps;
        void* base_addr;
        if (!m_allocator->allocate(nbytes, base_addr, deps)) {
            return false;
        }

        auto block = std::make_unique<Block>(base_addr, nbytes, std::move(deps));
        region = block->head.get();
        m_blocks.push_back(std::move(block));
    } else {
        region = *it;
        m_free_regions.erase(it);
    }

    auto padding = round_up_to_multiple(region->offset_in_block, alignment) - region->offset_in_block;

    if (padding > 0) {
        auto [left, right] = split_region(region, padding);
        m_free_regions.emplace(left);
        region = right;
    }

    if (nbytes != region->size) {
        auto [left, right] = split_region(region, nbytes);
        m_free_regions.emplace(right);
        region = left;
    }

    auto* block = region->parent;
    addr_out = static_cast<char*>(block->base_addr) + region->offset_in_block;
    deps_out.insert(region->dependencies);
    m_used_regions.emplace(addr_out, region);
    return true;
}

void PoolAllocator::deallocate(void* addr, size_t nbytes, DeviceEventSet deps) {
    auto it = m_used_regions.find(addr);
    KMM_ASSERT(it != m_used_regions.end());

    auto* region = it->second;
    m_used_regions.erase(it);

    KMM_ASSERT(nbytes <= region->size);
    KMM_ASSERT(region->in_use == true);

    region->in_use = false;
    region->dependencies.insert(std::move(deps));

    auto* next = region->next.get();
    auto* prev = region->prev;

    if (prev != nullptr && !prev->in_use) {
    }

    if (next != nullptr && !next->in_use) {
    }
}

auto PoolAllocator::split_region(Region* region, size_t left_size) -> std::pair<Region*, Region*>{
    KMM_TODO();
}

}


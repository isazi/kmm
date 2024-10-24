#include "kmm/allocators/block.hpp"
#include "kmm/utils/integer_fun.hpp"

namespace kmm {

BlockAllocator::BlockAllocator(std::unique_ptr<AsyncAllocator> allocator, size_t min_block_size) :
    m_allocator(std::move(allocator)),
    m_min_block_size(min_block_size) {}

BlockAllocator::~BlockAllocator() {
    KMM_TODO();
}

struct BlockAllocator::Block {
    std::set<Region*, RegionSizeCompare> free_regions;
    std::unique_ptr<Region> head = nullptr;
    Region* tail = nullptr;
    void* base_addr = nullptr;
    size_t size = 0;

    Block(void* addr, size_t size, DeviceEventSet deps) {
        this->head = std::make_unique<Region>(this, 0, size, (deps));
        this->tail = head.get();
        this->free_regions.insert(this->head.get());
        this->base_addr = addr;
        this->size = size;
    }
};

struct BlockAllocator::Region {
    Block* parent = nullptr;
    std::unique_ptr<Region> next = nullptr;
    Region* prev = nullptr;
    size_t offset_in_block = 0;
    size_t size = 0;
    bool is_free = false;
    DeviceEventSet dependencies;

    Region(Block* parent, size_t offset_in_block, size_t size, DeviceEventSet deps = {}) {
        this->parent = parent;
        this->offset_in_block = offset_in_block;
        this->size = size;
        this->dependencies = (deps);
    }
};

bool BlockAllocator::RegionSizeCompare::operator()(const Region* a, const Region* b) const {
    return a->size < b->size;
}

bool BlockAllocator::RegionSizeCompare::operator()(const Region* a, RegionSize b) const {
    return a->size < b.value;
}

static constexpr size_t MAX_ALIGNMENT = 256;

bool BlockAllocator::allocate_async(size_t nbytes, void** addr_out, DeviceEventSet* deps_out) {
    size_t alignment = std::min(round_up_to_power_of_two(nbytes), MAX_ALIGNMENT);
    nbytes = round_up_to_multiple(nbytes, alignment);

    Region* region = nullptr;
    for (size_t i = 0; i < m_blocks.size(); i++) {
        auto* block = m_blocks[m_active_block].get();
        auto it = block->free_regions.lower_bound(RegionSize {nbytes});

        while (it != block->free_regions.end()) {
            auto& r = **it;
            auto padding = round_up_to_multiple(r.offset_in_block, alignment) - r.offset_in_block;

            if (r.size >= padding + nbytes) {
                region = &r;
                break;
            }
        }

        if (region != nullptr) {
            break;
        }

        m_active_block = (m_active_block + 1) % m_blocks.size();
    }

    if (region == nullptr) {
        DeviceEventSet deps;
        void* base_addr;
        size_t block_size = std::max(nbytes, m_min_block_size);

        if (!m_allocator->allocate_async(block_size, &base_addr, &deps)) {
            return false;
        }

        auto new_block = std::make_unique<Block>(base_addr, block_size, std::move(deps));
        region = new_block->head.get();

        m_blocks.insert(m_blocks.begin() + m_active_block, std::move(new_block));
    }

    auto* block = region->parent;
    block->free_regions.erase(region);

    auto padding =
        round_up_to_multiple(region->offset_in_block, alignment) - region->offset_in_block;

    if (padding > 0) {
        auto [left, right] = split_region(region, padding);
        block->free_regions.emplace(left);
        region = right;
    }

    if (region->size != nbytes) {
        auto [left, right] = split_region(region, nbytes);
        block->free_regions.emplace(right);
        region = left;
    }

    region->is_free = false;
    m_active_regions.emplace(addr_out, region);

    *addr_out = static_cast<char*>(block->base_addr) + region->offset_in_block;
    deps_out->insert(region->dependencies);
    return true;
}

void BlockAllocator::deallocate_async(void* addr, size_t nbytes, DeviceEventSet deps) {
    auto it = m_active_regions.find(addr);
    KMM_ASSERT(it != m_active_regions.end());

    auto* region = it->second;
    m_active_regions.erase(it);

    KMM_ASSERT(nbytes <= region->size);
    KMM_ASSERT(region->is_free == false);

    region->is_free = true;
    region->dependencies = std::move(deps);

    auto* block = region->parent;
    auto* prev = region->prev;
    auto* next = region->next.get();

    if (prev != nullptr && prev->is_free) {
        block->free_regions.erase(prev);
        region = merge_regions(prev, region);
    }

    if (next != nullptr && next->is_free) {
        block->free_regions.erase(next);
        region = merge_regions(region, next);
    }

    block->free_regions.insert(region);
}

auto BlockAllocator::split_region(Region* region, size_t left_size) -> std::pair<Region*, Region*> {
    KMM_ASSERT(region->size > left_size);

    auto* parent = region->parent;
    auto* left = region;

    size_t right_offset = left->offset_in_block + left_size;
    size_t right_size = left->size - left_size;
    left->size = left_size;

    auto right = std::make_unique<Region>(parent, right_offset, right_size, left->dependencies);

    if (left->next != nullptr) {
        left->next->prev = right.get();
        right->next = std::move(left->next);
    } else {
        parent->tail = right.get();
        right->next = nullptr;
    }

    auto* right_ptr = right.get();
    right->prev = left;
    left->next = std::move(right);

    return {left, right_ptr};
}

auto BlockAllocator::merge_regions(Region* left, Region* right) -> Region* {
    auto* parent = left->parent;

    KMM_ASSERT(left->parent == parent && right->parent == parent);
    KMM_ASSERT(left->is_free && right->is_free);
    KMM_ASSERT(left->next.get() == right);
    KMM_ASSERT(left == right->prev);

    if (right->next != nullptr) {
        right->next->prev = left;
    } else {
        parent->tail = left;
    }

    left->size += right->size;
    left->dependencies.insert(right->dependencies);
    left->next = std::move(right->next);  // `right` is deleted here (since left.next == right)
    return left;
}

void BlockAllocator::make_progress() {
    m_allocator->make_progress();
}

void BlockAllocator::trim(size_t nbytes_remaining) {
    size_t nbytes_allocated = 0;

    for (const auto& block : m_blocks) {
        nbytes_allocated += block->size;
    }

    while (nbytes_allocated < nbytes_remaining) {
        size_t index = 0;
        bool found = false;

        while (index < m_blocks.size()) {
            auto* region = m_blocks[index]->head.get();

            if (region->is_free && region->next == nullptr) {
                found = true;
                break;
            }
        }

        if (!found) {
            break;
        }

        auto& block = m_blocks[index];
        auto* region = block->head.get();
        block->free_regions.erase(region);

        m_allocator->deallocate_async(  //
            block->base_addr,
            block->size,
            std::move(region->dependencies)
        );

        nbytes_allocated -= region->size;
        m_blocks.erase(m_blocks.begin() + static_cast<ptrdiff_t>(index));

        if (m_active_block > index) {
            m_active_block--;
        }
    }

    m_allocator->trim(nbytes_remaining);
}

}  // namespace kmm

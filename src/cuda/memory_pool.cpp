#include <set>

#include "kmm/cuda/memory_pool.hpp"
#include "kmm/panic.hpp"

namespace kmm {

static size_t calculate_padding_for_alignment(size_t offset, size_t alignment) {
    if ((offset & (alignment - 1)) == 0) {
        return 0;
    }

    return alignment - (offset & (alignment - 1));
}

struct MemoryPool::BlockRange {
    BlockRange(Block* parent, uintptr_t addr, size_t size) :
        parent(parent),
        addr(addr),
        size(size) {}
    BlockRange(const BlockRange&) = delete;

    Block* parent;
    bool is_free = true;
    uintptr_t addr = 0;
    size_t size = 0;
    BlockRange* prev = nullptr;
    BlockRange* next = nullptr;
};

struct MemoryPool::BlockSize {
    size_t size;
};

struct MemoryPool::BlockSizeCompare {
    using is_transparent = void;

    bool operator()(const BlockRange* lhs, const BlockRange* rhs) const {
        return lhs->size < rhs->size || (lhs->size == rhs->size && lhs->addr < rhs->addr);
    }

    bool operator()(const std::unique_ptr<BlockRange>& lhs, const std::unique_ptr<BlockRange>& rhs)
        const {
        return (*this)(lhs.get(), rhs.get());
    }

    bool operator()(const std::unique_ptr<BlockRange>& lhs, const BlockRange* rhs) const {
        return (*this)(lhs.get(), rhs);
    }

    bool operator()(const BlockRange* lhs, const std::unique_ptr<BlockRange>& rhs) const {
        return (*this)(lhs, rhs.get());
    }

    bool operator()(const std::unique_ptr<BlockRange>& lhs, BlockSize rhs) const {
        return lhs->size < rhs.size;
    }

    bool operator()(BlockSize lhs, const std::unique_ptr<BlockRange>& rhs) const {
        return lhs.size < rhs->size;
    }
};

struct MemoryPool::Block {
    Block(uintptr_t addr, size_t size) : base_addr(addr), size(size) {}
    Block(const Block&) = delete;

    uintptr_t base_addr;
    size_t size;
    size_t bytes_in_use = 0;
    std::set<std::unique_ptr<BlockRange>, BlockSizeCompare> free_blocks;
};

MemoryPool::MemoryPool() = default;
MemoryPool::~MemoryPool() = default;

void MemoryPool::insert_block(void* addr, size_t size) {
    auto block = std::make_unique<Block>(reinterpret_cast<uintptr_t>(addr), size);

    block->free_blocks.emplace(
        std::make_unique<BlockRange>(block.get(), block->base_addr, block->size));

    m_blocks.insert(m_blocks.begin() + m_active_block, std::move(block));
}

bool MemoryPool::remove_empty_block(void*& addr_out, size_t& size_out) {
    for (size_t i = 0; i < m_blocks.size(); i++) {
        if (m_blocks[i]->bytes_in_use > 0) {
            continue;
        }

        auto block = std::move(m_blocks[i]);
        m_blocks.erase(m_blocks.begin() + i);

        if (i > m_active_block) {
            m_active_block -= 1;
        }

        addr_out = reinterpret_cast<void*>(block->base_addr);
        size_out = block->size;
        return true;
    }

    return false;
}

void* MemoryPool::allocate_range(size_t alloc_size, size_t alloc_align) {
    for (size_t i = 0; i < m_blocks.size(); i++) {
        auto& block = *m_blocks[m_active_block];
        auto gap = remove_free_range_from_block(block, alloc_size, alloc_align);

        if (!gap) {
            m_active_block = (m_active_block + 1) % m_blocks.size();
            continue;
        }

        size_t gap_before = calculate_padding_for_alignment(gap->addr, alloc_align);

        if (gap_before > 0) {
            insert_free_range_into_block(  //
                block,
                gap->addr,
                gap_before,
                gap->prev,
                gap.get());
        }

        size_t gap_after = gap->size - gap_before - alloc_size;

        if (gap_after > 0) {
            insert_free_range_into_block(
                block,
                gap->addr + gap_before + alloc_size,
                gap_after,
                gap.get(),
                gap->next);
        }

        gap->is_free = false;
        gap->addr += gap_before;
        gap->size -= gap_before + gap_after;

        gap->parent->bytes_in_use += gap->size;

        void* addr = reinterpret_cast<void*>(gap->addr);
        m_allocated.emplace(uintptr_t(addr), std::move(gap));
        return addr;
    }

    return nullptr;
}

size_t MemoryPool::deallocate_range(void* addr) {
    auto it = m_allocated.find(uintptr_t(addr));
    KMM_ASSERT(it != m_allocated.end());

    auto range = std::move(it->second);
    m_allocated.erase(it);
    range->is_free = true;
    size_t range_size = range->size;

    auto& block = range->parent;
    block->bytes_in_use -= range->size;

    auto* old_prev = range->prev;
    if (old_prev != nullptr && old_prev->is_free) {
        range->addr -= old_prev->size;
        range->size += old_prev->size;

        if (auto* new_prev = old_prev->prev) {
            range->prev = new_prev;
            new_prev->next = range.get();
        } else {
            range->prev = nullptr;
        }

        auto m = block->free_blocks.find(old_prev);
        KMM_ASSERT(m->get() == old_prev);
        block->free_blocks.erase(m);
    }

    auto* old_next = range->next;
    if (old_next != nullptr && old_next->is_free) {
        range->size += old_next->size;

        if (auto* new_next = old_next->next) {
            range->next = new_next;
            new_next->prev = range.get();
        } else {
            range->next = nullptr;
        }

        auto m = block->free_blocks.find(old_next);
        KMM_ASSERT(m->get() == old_next);
        block->free_blocks.erase(m);
    }

    block->free_blocks.insert(std::move(range));
    return range_size;
}

size_t MemoryPool::num_blocks() const {
    return m_blocks.size();
}

void MemoryPool::insert_free_range_into_block(
    Block& parent,
    size_t addr,
    size_t size,
    BlockRange* prev,
    BlockRange* next) {
    auto new_range = std::make_unique<BlockRange>(&parent, addr, size);

    if (prev != nullptr) {
        prev->next = new_range.get();
        new_range->prev = prev;
    }

    if (next != nullptr) {
        next->prev = new_range.get();
        new_range->next = next;
    }

    parent.free_blocks.emplace(std::move(new_range));
}

std::unique_ptr<MemoryPool::BlockRange> MemoryPool::remove_free_range_from_block(
    Block& block,
    size_t size,
    size_t align) {
    auto it = block.free_blocks.lower_bound(BlockSize {size});

    while (it != block.free_blocks.end()) {
        const auto& range = *it;

        if (!range->is_free) {
            it++;
            continue;
        }

        size_t padding = calculate_padding_for_alignment(range->addr, align);

        if (range->size < padding + size) {
            it++;
            continue;
        }

        auto node = block.free_blocks.extract(it);
        return std::move(node.value());
    }

    return nullptr;
}

}  // namespace kmm
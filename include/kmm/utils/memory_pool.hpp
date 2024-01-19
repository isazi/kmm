#include <memory>
#include <unordered_map>
#include <vector>

namespace kmm {

class MemoryPool {
    struct Block;
    struct BlockRange;
    struct BlockSize;
    struct BlockSizeCompare;

  public:
    MemoryPool();
    ~MemoryPool();

    void insert_block(void* addr, size_t size);
    bool remove_empty_block(void** addr_out, size_t* size_out);
    void* allocate_range(size_t alloc_size, size_t alloc_align);
    size_t deallocate_range(void* addr);
    size_t num_blocks() const;

  public:
    static void insert_free_range_into_block(
        Block* parent,
        size_t addr,
        size_t size,
        BlockRange* prev,
        BlockRange* next);

    static std::unique_ptr<BlockRange> remove_free_range_from_block(
        Block* parent,
        size_t size,
        size_t align);

    size_t m_active_block = 0;
    std::vector<std::unique_ptr<Block>> m_blocks;
    std::unordered_map<uintptr_t, std::unique_ptr<BlockRange>> m_allocated;
};

}  // namespace kmm

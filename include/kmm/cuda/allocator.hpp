#include "memory_pool.hpp"
#include "types.hpp"

namespace kmm {

/**
 * Base class for `CudaPinnedAllocator` and `CudaDeviceAllocator`
 */
class CudaAllocatorBase {
  public:
    static constexpr size_t MAX_ALIGN = 128;
    static constexpr size_t DEFAULT_BLOCK_SIZE = 524'288'000;

    /**
     * Create a new CUDA allocator. This allocator uses a caching strategy where large `blocks' of
     * physical memory are allocated from the underlying allocator and those large blocks are then
     * split into smaller allocations.
     *
     * @param max_bytes Set a limit on how many bytes may be allocated at once.
     * @param block_size The block size in bytes.
     */
    CudaAllocatorBase(size_t max_bytes, size_t block_size = DEFAULT_BLOCK_SIZE);
    virtual ~CudaAllocatorBase();

    /**
     * Allocate a range of `size` bytes. Returns NULL on failure.
     *
     * @param size The size of the range in bytes.
     * @param align The alignment of the allocation. Must be power of two of at most `MAX_ALIGN`.
     * @return A pointer to the data.
     */
    void* allocate(size_t size, size_t align);

    /**
     * Deallocate the memory range that starts at `addr`.
     *
     * @param addr
     */
    void deallocate(void* addr);

    /**
     * This allocator uses a cache strategy that reuses previous allocations. These methods will
     * free some or all the allocated memory that is not in use at the moment.
     */
    size_t reclaim_some_free_memory();
    void reclaim_free_memory();

    /**
     * Returns how many bytes are currently in use.
     */
    size_t bytes_in_use() const {
        return m_bytes_inuse;
    }

  private:
    virtual void* allocate_impl(size_t size) = 0;
    virtual void deallocate_impl(void* addr) = 0;

    size_t m_block_size;
    size_t m_bytes_inuse = 0;
    size_t m_current_capacity = 0;
    size_t m_max_capacity = 0;
    MemoryPool m_pool;
};

/**
 * Allocator that calls `cuMemHostAlloc` to allocate memory.
 */
class CudaPinnedAllocator final: CudaAllocatorBase {
    CudaPinnedAllocator(
        CudaContextHandle context,
        size_t max_bytes,
        size_t block_size = DEFAULT_BLOCK_SIZE);
    void* allocate_impl(size_t size) final;
    void deallocate_impl(void* addr) final;

  private:
    CudaContextHandle m_context;
};

/**
 * Allocator that calls `cuMemAlloc` to allocate memory.
 */
class CudaDeviceAllocator final: CudaAllocatorBase {
    CudaDeviceAllocator(CudaContextHandle context);
    CudaDeviceAllocator(
        CudaContextHandle context,
        size_t max_bytes,
        size_t block_size = DEFAULT_BLOCK_SIZE);
    void* allocate_impl(size_t size) final;
    void deallocate_impl(void* addr) final;

  private:
    CudaContextHandle m_context;
};

}  // namespace kmm
#pragma once

#include "kmm/cuda/types.hpp"
#include "kmm/utils/memory_pool.hpp"

#ifdef KMM_USE_CUDA

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
     * @param max_bytes Set a limit on the total number of bytes that may be allocated.
     * @param block_size The block size in bytes. The default is 500MB.
     */
    CudaAllocatorBase(size_t max_bytes, size_t block_size = DEFAULT_BLOCK_SIZE);
    virtual ~CudaAllocatorBase();

    /**
     * Try to allocate a range of `size` bytes. Returns NULL on failure.
     *
     * @param size The size of the range in bytes.
     * @param align The alignment of the allocation. Must be power of two and at most `MAX_ALIGN`.
     * @return A pointer to the data.
     */
    void* allocate(size_t size, size_t align = 1);

    /**
     * Deallocate the memory range that starts at `addr`.
     *
     * @param addr
     */
    void deallocate(void* addr);

    /**
     * This allocator uses a cache strategy that reuses previous allocations. This method will
     * free all of the allocated memory that is not in use at the moment.
     */
    void reclaim_free_memory();

    /**
     * This allocator uses a cache strategy that reuses previous allocations. This methods will
     * try to free some of the allocated memory that is not in use. It returns `true` if some
     * memory was freed and `false` if no memory could be freed.
     */
    bool try_reclaim_some_free_memory();

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
class CudaPinnedAllocator final: public CudaAllocatorBase {
  public:
    CudaPinnedAllocator(
        CudaContextHandle context,
        size_t max_bytes,
        size_t block_size = DEFAULT_BLOCK_SIZE);

  private:
    void* allocate_impl(size_t size) final;
    void deallocate_impl(void* addr) final;

    CudaContextHandle m_context;
};

/**
 * Allocator that calls `cuMemAlloc` to allocate memory.
 */
class CudaDeviceAllocator final: public CudaAllocatorBase {
  public:
    CudaDeviceAllocator(CudaContextHandle context);
    CudaDeviceAllocator(
        CudaContextHandle context,
        size_t max_bytes,
        size_t block_size = DEFAULT_BLOCK_SIZE);

  private:
    void* allocate_impl(size_t size) final;
    void deallocate_impl(void* addr) final;

    CudaContextHandle m_context;
};

}  // namespace kmm

#endif  // KMM_USE_CUDA
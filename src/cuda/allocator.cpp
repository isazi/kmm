#include "spdlog/spdlog.h"

#include "kmm/cuda/allocator.hpp"
#include "kmm/panic.hpp"

namespace kmm {

static size_t round_up_to_power_of_two(size_t size) {
    size_t align = 1;
    while (align < size && align <= std::numeric_limits<size_t>::max() / 2) {
        align *= 2;
    }

    return align;
}

static size_t round_up_to_multiple(size_t n, size_t k) {
    return n + (n % k == 0 ? 0 : k - n % k);
}

CudaAllocatorBase::CudaAllocatorBase(size_t max_bytes, size_t block_size) :
    m_block_size(block_size),
    m_max_capacity(max_bytes) {}

CudaAllocatorBase::~CudaAllocatorBase() {
    reclaim_free_memory();

    if (m_current_capacity > 0) {
        KMM_PANIC(
            "A cuda allocator was deleted while there where still {} bytes in use and {} bytes allocated",
            m_bytes_inuse,
            m_current_capacity);
    }
}

void* CudaAllocatorBase::allocate(size_t size, size_t align) {
    align = std::min(round_up_to_power_of_two(std::max(size, align)), MAX_ALIGN);
    size_t aligned_size = round_up_to_multiple(size, align);

    // Try to allocate. If successful, we are done.
    if (auto* ptr = m_pool.allocate_range(aligned_size, align)) {
        m_bytes_inuse += aligned_size;
        return ptr;
    }

    // The blocks must be:
    // - at least large enough for `aligned_size` bytes
    // - at most `m_bytes_capacity - m_bytes_allocated` to ensure we do not exceed capacity
    size_t min_block_size = aligned_size + align;
    size_t max_block_size = m_max_capacity - m_current_capacity;
    size_t new_block_size = std::max(std::min(m_block_size, max_block_size), min_block_size);

    if (new_block_size > max_block_size) {
        return nullptr;
    }

    // Try to allocate the block.  While we fail to allocate a block, deallocate an empty block
    // and retry block allocation.
    while (true) {
        // Try to allocate. If success, insert the block
        if (auto* block_addr = allocate_impl(new_block_size)) {
            m_current_capacity += new_block_size;
            m_pool.insert_block(block_addr, new_block_size);
            break;
        }

        // Otherwise, reclaim some free memory and try again
        if (reclaim_some_free_memory() > 0) {
            return nullptr;
        }
    }

    // Should never fail since we just inserted a block of the right size.
    auto* ptr = m_pool.allocate_range(aligned_size, align);
    KMM_ASSERT(ptr != nullptr);
    m_bytes_inuse += aligned_size;

    return ptr;
}

void CudaAllocatorBase::deallocate(void* addr) {
    size_t size = m_pool.deallocate_range(addr);
    m_bytes_inuse -= size;
}

size_t CudaAllocatorBase::reclaim_some_free_memory() {
    void* addr;
    size_t size;

    if (!m_pool.remove_empty_block(addr, size)) {
        return 0;
    }

    m_current_capacity -= size;
    deallocate_impl(addr);
    return size;
}

void CudaAllocatorBase::reclaim_free_memory() {
    while (reclaim_some_free_memory() > 0) {
    }
}

CudaPinnedAllocator::CudaPinnedAllocator(
    CudaContextHandle context,
    size_t max_bytes,
    size_t block_size) :
    CudaAllocatorBase(max_bytes, block_size),
    m_context(std::move(context)) {}

void* CudaPinnedAllocator::allocate_impl(size_t size) {
    CudaContextGuard guard {m_context};
    void* ptr = nullptr;
    unsigned int flags = CU_MEMHOSTALLOC_PORTABLE;
    auto result = cuMemHostAlloc(&ptr, size, flags);

    if (result == CUDA_SUCCESS) {
        return ptr;
    }

    if (result != CUDA_ERROR_OUT_OF_MEMORY) {
        const char* message = "";
        cuGetErrorString(result, &message);
        spdlog::warn("allocation of {} bytes failed on host: {}", size, message);
    }

    return nullptr;
}

void CudaPinnedAllocator::deallocate_impl(void* addr) {
    CudaContextGuard guard {m_context};
    KMM_CUDA_CHECK(cuMemFreeHost(addr));
}

CudaDeviceAllocator::CudaDeviceAllocator(
    CudaContextHandle context,
    size_t max_bytes,
    size_t block_size) :
    CudaAllocatorBase(max_bytes, block_size),
    m_context(std::move(context)) {}

size_t determine_default_device_capacity(CudaContextHandle& context) {
    // Keep 250MB as a margin
    static constexpr size_t MARGIN = 250'000'000;

    CudaContextGuard guard {context};
    size_t bytes_free;
    size_t bytes_total;
    KMM_CUDA_CHECK(cuMemGetInfo(&bytes_free, &bytes_total));

    return bytes_free > MARGIN ? bytes_free - MARGIN : 0;
}

CudaDeviceAllocator::CudaDeviceAllocator(CudaContextHandle context) :
    CudaDeviceAllocator(context, determine_default_device_capacity(context)) {}

void* CudaDeviceAllocator::allocate_impl(size_t size) {
    CudaContextGuard guard {m_context};

    CUdeviceptr ptr = 0xdeadbeef;
    auto result = cuMemAlloc(&ptr, size);

    if (result == CUDA_SUCCESS) {
        return reinterpret_cast<void*>(ptr);
    }

    if (result != CUDA_ERROR_OUT_OF_MEMORY) {
        const char* message = "";
        cuGetErrorString(result, &message);
        spdlog::warn("allocation of {} bytes failed on device: {}", size, message);
    }

    return nullptr;
}

void CudaDeviceAllocator::deallocate_impl(void* addr) {
    CudaContextGuard guard {m_context};
    KMM_CUDA_CHECK(cuMemFree(reinterpret_cast<CUdeviceptr>(addr)));
}
};  // namespace kmm

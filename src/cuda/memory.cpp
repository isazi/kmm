#include <deque>
#include <functional>
#include <utility>

#include "kmm/cuda/copy_engine.hpp"
#include "kmm/cuda/memory.hpp"
#include "kmm/cuda/types.hpp"
#include "kmm/host/memory.hpp"

#ifdef KMM_USE_CUDA

namespace kmm {

static constexpr MemoryId HOST_MEMORY = MemoryId(0);

static constexpr size_t memory_id_to_device_index(MemoryId id) {
    KMM_ASSERT(id.get() > 0);
    return id.get() - 1;
}

void CudaAllocation::copy_from_host_sync(
    const void* src_addr,
    size_t dst_offset,
    size_t num_bytes) {
    void* dst_addr = reinterpret_cast<char*>(m_data) + dst_offset;

    KMM_CUDA_CHECK(cuMemcpy(  //
        reinterpret_cast<CUdeviceptr>(dst_addr),
        reinterpret_cast<CUdeviceptr>(src_addr),
        num_bytes));
}

void CudaAllocation::copy_to_host_sync(size_t src_offset, void* dst_addr, size_t num_bytes) const {
    const void* src_addr = reinterpret_cast<const char*>(m_data) + src_offset;

    KMM_CUDA_CHECK(cuMemcpy(  //
        reinterpret_cast<CUdeviceptr>(dst_addr),
        reinterpret_cast<CUdeviceptr>(src_addr),
        num_bytes));
}

CudaMemory::CudaMemory(
    std::shared_ptr<ThreadPool> host_thread,
    std::vector<CudaContextHandle> contexts) :
    m_pool(std::move(host_thread)) {
    KMM_ASSERT(!contexts.empty());

    m_copy_engine = std::make_shared<CudaCopyEngine>(contexts);
    m_copy_thread = std::thread([engine = m_copy_engine]() { engine->run_forever(); });

    m_host_allocator = std::make_unique<CudaPinnedAllocator>(contexts[0], 40L * 524'288'000);

    for (const auto& context : contexts) {
        m_device_allocators.emplace_back(std::make_unique<CudaDeviceAllocator>(context));
    }
}

CudaMemory::~CudaMemory() {
    m_copy_engine->shutdown();
    m_copy_thread.join();

    m_host_allocator->reclaim_free_memory();
    for (auto& device : m_device_allocators) {
        device->reclaim_free_memory();
    }
}

std::optional<std::unique_ptr<MemoryAllocation>> CudaMemory::allocate(
    MemoryId id,
    size_t num_bytes) {
    if (id == HOST_MEMORY) {
        if (auto* ptr = m_host_allocator->allocate(num_bytes)) {
            return std::make_unique<PinnedAllocation>(ptr, num_bytes);
        }

        return std::nullopt;
    }

    auto index = memory_id_to_device_index(id);
    if (index < m_device_allocators.size()) {
        if (auto* ptr = m_device_allocators[index]->allocate(num_bytes)) {
            return std::make_unique<CudaAllocation>(ptr, num_bytes);
        }

        return std::nullopt;
    }

    return std::nullopt;
}

void CudaMemory::deallocate(MemoryId id, std::unique_ptr<MemoryAllocation> allocation) {
    if (id == HOST_MEMORY) {
        const auto* alloc = dynamic_cast<const HostAllocation*>(allocation.get());
        KMM_ASSERT(alloc != nullptr);

        m_host_allocator->deallocate(alloc->data());
        return;
    }

    auto index = memory_id_to_device_index(id);
    KMM_ASSERT(index < m_device_allocators.size());

    const auto* alloc = dynamic_cast<const CudaAllocation*>(allocation.get());
    KMM_ASSERT(alloc != nullptr);

    m_device_allocators[index]->deallocate(alloc->data());
}

void CudaMemory::copy_async(
    MemoryId src_id,
    const MemoryAllocation* src_alloc,
    size_t src_offset,
    MemoryId dst_id,
    const MemoryAllocation* dst_alloc,
    size_t dst_offset,
    size_t num_bytes,
    Completion completion) {
    const void* src_ptr = unpack_allocation(src_alloc, src_offset, num_bytes, src_id);
    void* dst_ptr = unpack_allocation(dst_alloc, dst_offset, num_bytes, dst_id);

    if (src_id == HOST_MEMORY && dst_id == HOST_MEMORY) {
        m_pool->submit_copy(src_ptr, dst_ptr, num_bytes, std::move(completion));
    } else if (src_id == HOST_MEMORY) {
        m_copy_engine->copy_host_to_device_async(
            memory_id_to_device_index(dst_id),
            src_ptr,
            dst_ptr,
            num_bytes,
            std::move(completion));
    } else if (dst_id == HOST_MEMORY) {
        m_copy_engine->copy_device_to_host_async(
            memory_id_to_device_index(src_id),
            src_ptr,
            dst_ptr,
            num_bytes,
            std::move(completion));
    } else {
        m_copy_engine->copy_device_to_device_async(
            memory_id_to_device_index(src_id),
            memory_id_to_device_index(dst_id),
            src_ptr,
            dst_ptr,
            num_bytes,
            std::move(completion));
    }
}

void* CudaMemory::unpack_allocation(
    const MemoryAllocation* raw_alloc,
    size_t offset,
    size_t num_bytes,
    MemoryId memory_id) {
    if (memory_id == HOST_MEMORY) {
        const auto* alloc = dynamic_cast<const HostAllocation*>(raw_alloc);
        KMM_ASSERT(alloc != nullptr && alloc->size() >= offset + num_bytes);
        return static_cast<char*>(alloc->data()) + offset;
    } else {
        const auto* alloc = dynamic_cast<const CudaAllocation*>(raw_alloc);
        KMM_ASSERT(alloc != nullptr && alloc->size() >= offset + num_bytes);
        return static_cast<char*>(alloc->data()) + offset;
    }
}

bool CudaMemory::is_copy_possible(MemoryId src_id, MemoryId dst_id) {
    return src_id == HOST_MEMORY || dst_id == HOST_MEMORY || src_id == dst_id;
}

void CudaMemory::fill_async(
    MemoryId dst_id,
    const MemoryAllocation* dst_alloc,
    size_t dst_offset,
    size_t num_bytes,
    std::vector<uint8_t> fill_pattern,
    Completion completion) {
    if (dst_id == HOST_MEMORY) {
        const auto* alloc = dynamic_cast<const HostAllocation*>(dst_alloc);
        KMM_ASSERT(alloc != nullptr);
        KMM_ASSERT(alloc->size() >= dst_offset + num_bytes);

        m_pool->submit_fill(
            static_cast<char*>(alloc->data()) + dst_offset,
            num_bytes,
            std::move(fill_pattern),
            std::move(completion));
    } else {
        const auto* alloc = dynamic_cast<const CudaAllocation*>(dst_alloc);
        KMM_ASSERT(alloc != nullptr);
        KMM_ASSERT(alloc->size() >= dst_offset + num_bytes);

        m_copy_engine->fill_device_async(
            memory_id_to_device_index(dst_id),
            static_cast<char*>(alloc->data()) + dst_offset,
            num_bytes,
            std::move(fill_pattern),
            std::move(completion));
    }
}

}  // namespace kmm

#endif  // KMM_USE_CUDA
#include "kmm/allocators/device.hpp"

namespace kmm {

PinnedMemoryAllocator::PinnedMemoryAllocator(
    CudaContextHandle context,
    std::shared_ptr<CudaStreamManager> streams,
    size_t max_bytes
) :
    SyncAllocator(streams, max_bytes),
    m_context(context) {}

bool PinnedMemoryAllocator::allocate(size_t nbytes, void** addr_out) {
    CudaContextGuard guard {m_context};
    CUresult result =
        cuMemHostAlloc(addr_out, nbytes, CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP);

    if (result == CUDA_SUCCESS) {
        return true;
    } else if (result == CUDA_ERROR_OUT_OF_MEMORY) {
        return false;
    } else {
        throw CudaDriverException("error when calling `cuMemHostAlloc`", result);
    }
}

void PinnedMemoryAllocator::deallocate(void* addr, size_t nbytes) {
    CudaContextGuard guard {m_context};
    KMM_CUDA_CHECK(cuMemFreeHost(addr));
}

DeviceMemoryAllocator::DeviceMemoryAllocator(
    CudaContextHandle context,
    std::shared_ptr<CudaStreamManager> streams,
    size_t max_bytes
) :
    SyncAllocator(streams, max_bytes),
    m_context(context) {}

bool DeviceMemoryAllocator::allocate(size_t nbytes, void** addr_out) {
    CudaContextGuard guard {m_context};
    CUdeviceptr ptr;
    CUresult result = cuMemAlloc(&ptr, nbytes);

    if (result == CUDA_SUCCESS) {
        *addr_out = (void*)ptr;
        return true;
    } else if (result == CUDA_ERROR_OUT_OF_MEMORY) {
        return false;
    } else {
        throw CudaDriverException("error when calling `cuMemAlloc`", result);
    }
}

void DeviceMemoryAllocator::deallocate(void* addr, size_t nbytes) {
    CudaContextGuard guard {m_context};
    KMM_CUDA_CHECK(cuMemFree(CUdeviceptr(addr)));
}

struct DevicePoolAllocator::Allocation {
    void* addr;
    size_t nbytes;
    DeviceEvent event;
};

DevicePoolAllocator::DevicePoolAllocator(
    CudaContextHandle context,
    std::shared_ptr<CudaStreamManager> streams,
    DevicePoolKind kind,
    size_t max_bytes
) :
    m_context(context),
    m_streams(streams),
    m_alloc_stream(streams->create_stream(context)),
    m_dealloc_stream(streams->create_stream(context)),
    m_kind(kind),
    m_bytes_limit(max_bytes) {
    CudaContextGuard guard {m_context};

    CUdevice device;
    KMM_CUDA_CHECK(cuCtxGetDevice(&device));

    switch (m_kind) {
        case DevicePoolKind::Default:
            KMM_CUDA_CHECK(cuDeviceGetDefaultMemPool(&m_pool, device));
            break;

        case DevicePoolKind::Create:
            CUmemPoolProps props;
            ::bzero(&props, sizeof(CUmemPoolProps));

            props.allocType = CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
            props.handleTypes = CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE;
            props.location.type = CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
            props.location.id = device;

            KMM_CUDA_CHECK(cuMemPoolCreate(&m_pool, &props));
            break;
    }
}

DevicePoolAllocator::~DevicePoolAllocator() {
    for (auto d : m_pending_deallocs) {
        m_bytes_in_use -= d.nbytes;
        m_streams->wait_until_ready(d.event);
    }

    KMM_ASSERT(m_bytes_in_use == 0);

    CudaContextGuard guard {m_context};

    switch (m_kind) {
        case DevicePoolKind::Default:
            // No need to destroy the default pool
            break;
        case DevicePoolKind::Create:
            KMM_CUDA_CHECK(cuMemPoolDestroy(m_pool));
            break;
    }
}

bool DevicePoolAllocator::allocate_async(size_t nbytes, void** addr_out, DeviceEventSet* deps_out) {
    make_progress();

    while (m_bytes_limit - m_bytes_in_use < nbytes) {
        if (m_pending_deallocs.empty()) {
            return false;
        }

        auto& d = m_pending_deallocs.front();
        m_streams->wait_for_event(m_alloc_stream, d.event);
        m_bytes_in_use -= d.nbytes;
        m_pending_deallocs.pop_front();
    }

    CUdeviceptr device_ptr;
    CUresult result = CUDA_ERROR_UNKNOWN;

    auto event = m_streams->with_stream(m_alloc_stream, [&](auto stream) {
        CudaContextGuard guard {m_context};
        result = cuMemAllocFromPoolAsync(&device_ptr, nbytes, m_pool, stream);
    });

    if (result == CUDA_SUCCESS) {
        m_bytes_in_use += nbytes;
        deps_out->insert(event);
        *addr_out = (void*)device_ptr;
        return true;
    } else if (result == CUDA_ERROR_OUT_OF_MEMORY) {
        return false;
    } else {
        throw CudaDriverException("error while calling `cuMemAllocFromPoolAsync`", result);
    }
}

void DevicePoolAllocator::deallocate_async(void* addr, size_t nbytes, DeviceEventSet deps) {
    CUdeviceptr device_ptr = (CUdeviceptr)addr;

    auto event = m_streams->with_stream(m_dealloc_stream, deps, [&](auto stream) {
        KMM_CUDA_CHECK(cuMemFreeAsync(device_ptr, stream));
    });

    m_pending_deallocs.push_back({.addr = addr, .nbytes = nbytes, .event = event});
}

void DevicePoolAllocator::make_progress() {
    while (true) {
        if (m_pending_deallocs.empty()) {
            break;
        }

        auto& d = m_pending_deallocs.front();

        if (!m_streams->is_ready(d.event)) {
            break;
        }

        m_bytes_in_use -= d.nbytes;
        m_pending_deallocs.pop_front();
    }
}

void DevicePoolAllocator::trim(size_t nbytes_remaining) {
    while (m_bytes_in_use > nbytes_remaining) {
        if (m_pending_deallocs.empty()) {
            break;
        }

        auto& d = m_pending_deallocs.front();
        m_streams->wait_until_ready(d.event);

        m_bytes_in_use -= d.nbytes;
        m_pending_deallocs.pop_front();
    }

    KMM_CUDA_CHECK(cuMemPoolTrimTo(m_pool, nbytes_remaining));
}
}  // namespace kmm
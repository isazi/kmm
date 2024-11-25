#include "kmm/allocators/device.hpp"

namespace kmm {

PinnedMemoryAllocator::PinnedMemoryAllocator(
    GPUContextHandle context,
    std::shared_ptr<DeviceStreamManager> streams,
    size_t max_bytes
) :
    SyncAllocator(streams, max_bytes),
    m_context(context) {}

bool PinnedMemoryAllocator::allocate(size_t nbytes, void** addr_out) {
    GPUContextGuard guard {m_context};
    GPUresult result =
        gpuMemHostAlloc(addr_out, nbytes, GPU_MEMHOSTALLOC_PORTABLE | GPU_MEMHOSTALLOC_DEVICEMAP);

    if (result == GPU_SUCCESS) {
        return true;
    } else if (result == GPU_ERROR_OUT_OF_MEMORY) {
        return false;
    } else {
        throw GPUDriverException("error when calling `cuMemHostAlloc`", result);
    }
}

void PinnedMemoryAllocator::deallocate(void* addr, size_t nbytes) {
    GPUContextGuard guard {m_context};
    KMM_GPU_CHECK(gpuMemFreeHost(addr));
}

DeviceMemoryAllocator::DeviceMemoryAllocator(
    GPUContextHandle context,
    std::shared_ptr<DeviceStreamManager> streams,
    size_t max_bytes
) :
    SyncAllocator(streams, max_bytes),
    m_context(context) {}

bool DeviceMemoryAllocator::allocate(size_t nbytes, void** addr_out) {
    GPUContextGuard guard {m_context};
    GPUdeviceptr ptr;
    GPUresult result = gpuMemAlloc(&ptr, nbytes);

    if (result == GPU_SUCCESS) {
        *addr_out = (void*)ptr;
        return true;
    } else if (result == GPU_ERROR_OUT_OF_MEMORY) {
        return false;
    } else {
        throw GPUDriverException("error when calling `cuMemAlloc`", result);
    }
}

void DeviceMemoryAllocator::deallocate(void* addr, size_t nbytes) {
    GPUContextGuard guard {m_context};
    KMM_GPU_CHECK(gpuMemFree(GPUdeviceptr(addr)));
}



DevicePoolAllocator::DevicePoolAllocator(
    GPUContextHandle context,
    std::shared_ptr<DeviceStreamManager> streams,
    DevicePoolKind kind,
    size_t max_bytes
) :
    m_context(context),
    m_streams(streams),
    m_alloc_stream(streams->create_stream(context)),
    m_dealloc_stream(streams->create_stream(context)),
    m_kind(kind),
    m_bytes_limit(max_bytes) {
    GPUContextGuard guard {m_context};

    GPUdevice device;
    KMM_GPU_CHECK(gpuCtxGetDevice(&device));

    switch (m_kind) {
        case DevicePoolKind::Default:
            KMM_GPU_CHECK(gpuDeviceGetDefaultMemPool(&m_pool, device));
            break;

        case DevicePoolKind::Create:
            GPUmemPoolProps props;
            ::bzero(&props, sizeof(GPUmemPoolProps));

            props.allocType = GPUmemAllocationType::GPU_MEM_ALLOCATION_TYPE_PINNED;
            props.handleTypes = GPUmemAllocationHandleType::GPU_MEM_HANDLE_TYPE_NONE;
            props.location.type = GPUmemLocationType::GPU_MEM_LOCATION_TYPE_DEVICE;
            props.location.id = device;

            KMM_GPU_CHECK(gpuMemPoolCreate(&m_pool, &props));
            break;
    }
}

DevicePoolAllocator::~DevicePoolAllocator() {
    for (auto d : m_pending_deallocs) {
        m_bytes_in_use -= d.nbytes;
        m_streams->wait_until_ready(d.event);
    }

    KMM_ASSERT(m_bytes_in_use == 0);

    GPUContextGuard guard {m_context};

    switch (m_kind) {
        case DevicePoolKind::Default:
            // No need to destroy the default pool
            break;
        case DevicePoolKind::Create:
            KMM_GPU_CHECK(gpuMemPoolDestroy(m_pool));
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

    GPUdeviceptr device_ptr;
    GPUresult result = GPUresult(GPU_ERROR_UNKNOWN);

    auto event = m_streams->with_stream(m_alloc_stream, [&](auto stream) {
        GPUContextGuard guard {m_context};
        result = gpuMemAllocFromPoolAsync(&device_ptr, nbytes, m_pool, stream);
    });

    if (result == GPU_SUCCESS) {
        m_bytes_in_use += nbytes;
        deps_out->insert(event);
        *addr_out = (void*)device_ptr;
        return true;
    } else if (result == GPU_ERROR_OUT_OF_MEMORY) {
        return false;
    } else {
        throw GPUDriverException("error while calling `cuMemAllocFromPoolAsync`", result);
    }
}

void DevicePoolAllocator::deallocate_async(void* addr, size_t nbytes, DeviceEventSet deps) {
    GPUdeviceptr device_ptr = (GPUdeviceptr)addr;

    auto event = m_streams->with_stream(m_dealloc_stream, deps, [&](auto stream) {
        KMM_GPU_CHECK(gpuMemFreeAsync(device_ptr, stream));
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

    KMM_GPU_CHECK(gpuMemPoolTrimTo(m_pool, nbytes_remaining));
}
}  // namespace kmm
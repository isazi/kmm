#include <unordered_map>

#include "spdlog/spdlog.h"

#include "kmm/internals/memory_allocator.hpp"

namespace kmm {

struct MemoryAllocatorImpl::Device {
    KMM_NOT_COPYABLE(Device)

  public:
    CudaContextHandle context;
    CUmemoryPool memory_pool = nullptr;
    size_t bytes_in_use = 0;
    size_t bytes_limit = std::numeric_limits<size_t>::max();

    CudaStream h2d_stream;
    CudaStream d2h_stream;
    CudaStream h2d_hi_stream;  // high priority stream
    CudaStream d2h_hi_stream;  // high priority stream
    CudaStream alloc_stream;
    CudaStream dealloc_stream;

    std::unordered_map<CUdeviceptr, size_t> active_allocations;
    std::deque<std::pair<CudaEvent, size_t>> pending_deallocations;

    Device(DeviceId device_id, MemoryDeviceInfo info, CudaStreamManager& streams) :
        context(streams.get(device_id)),
        h2d_stream(streams.create_stream(device_id, false)),
        d2h_stream(streams.create_stream(device_id, false)),
        h2d_hi_stream(streams.create_stream(device_id, true)),
        d2h_hi_stream(streams.create_stream(device_id, true)),
        alloc_stream(streams.create_stream(device_id)),
        dealloc_stream(streams.create_stream(device_id)),
        bytes_limit(info.num_bytes_limit) {
        CudaContextGuard guard {context};

        CUdevice device;
        KMM_CUDA_CHECK(cuCtxGetDevice(&device));

        size_t total_bytes;
        size_t free_bytes;
        KMM_CUDA_CHECK(cuMemGetInfo(&free_bytes, &total_bytes));

        bytes_limit = std::min(
            info.num_bytes_limit,
            saturating_sub(total_bytes, info.num_bytes_keep_available));

        CUmemPoolProps props;
        ::bzero(&props, sizeof(CUmemPoolProps));
        props.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
        props.handleTypes = CU_MEM_HANDLE_TYPE_NONE;
        props.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        props.location.id = device;

        KMM_CUDA_CHECK(cuMemPoolCreate(&memory_pool, &props));
    }

    Device(Device&& that) :
        context(that.context),
        h2d_stream(that.h2d_stream),
        d2h_stream(that.d2h_stream),
        h2d_hi_stream(that.h2d_hi_stream),
        d2h_hi_stream(that.d2h_hi_stream),
        alloc_stream(that.alloc_stream),
        dealloc_stream(that.dealloc_stream),
        bytes_limit(that.bytes_limit),
        bytes_in_use(that.bytes_in_use),
        memory_pool(that.memory_pool),
        active_allocations(std::move(that.active_allocations)) {
        that.memory_pool = nullptr;
        that.bytes_in_use = 0;
    }

    ~Device() {
        CudaContextGuard guard {context};
        KMM_ASSERT(bytes_in_use == 0);

        if (memory_pool != nullptr) {
            KMM_CUDA_CHECK(cuMemPoolDestroy(memory_pool));
        }
    }
};

struct MemoryAllocatorImpl::DeferredDeletion {
    CudaEventSet dependencies;
    void* data;
};

MemoryAllocatorImpl::MemoryAllocatorImpl(
    std::shared_ptr<CudaStreamManager> streams,
    std::vector<MemoryDeviceInfo> devices) :
    m_streams(streams) {
    for (size_t i = 0; i < devices.size(); i++) {
        m_devices.emplace_back(DeviceId(i), devices[i], *streams);
    }
}

MemoryAllocatorImpl::~MemoryAllocatorImpl() {
    for (auto& device : m_devices) {
        while (!device.pending_deallocations.empty()) {
            auto [dealloc_event, dealloc_size] = device.pending_deallocations.front();
            device.pending_deallocations.pop_front();

            m_streams->wait_until_ready(dealloc_event);
            device.bytes_in_use -= dealloc_size;
        }
    }
}

bool MemoryAllocatorImpl::allocate_device(
    DeviceId device_id,
    size_t nbytes,
    CUdeviceptr& ptr_out,
    CudaEvent& event_out) {
    auto& device = m_devices.at(device_id.get());

    while (device.bytes_limit - device.bytes_in_use < nbytes) {
        if (device.pending_deallocations.empty()) {
            return false;
        }

        auto [dealloc_event, dealloc_size] = device.pending_deallocations.front();
        device.pending_deallocations.pop_front();

        device.bytes_in_use -= dealloc_size;
        m_streams->wait_for_event(device.alloc_stream, dealloc_event);
    }

    CUresult result = CUDA_ERROR_UNKNOWN;

    event_out = m_streams->with_stream(device.alloc_stream, [&](auto stream) {
        result = cuMemAllocFromPoolAsync(&ptr_out, nbytes, device.memory_pool, stream);
    });

    if (result == CUDA_ERROR_OUT_OF_MEMORY) {
        return false;
    }

    if (result != CUDA_SUCCESS) {
        throw CudaDriverException("`cuMemAllocFromPoolAsync` failed", result);
    }

    device.bytes_in_use += nbytes;
    device.active_allocations.emplace(ptr_out, nbytes);
    return true;
}

void MemoryAllocatorImpl::deallocate_device(
    DeviceId device_id,
    CUdeviceptr ptr,
    CudaEventSet deps) {
    auto& device = m_devices.at(device_id.get());

    auto it = device.active_allocations.find(ptr);
    KMM_ASSERT(it != device.active_allocations.end());

    size_t nbytes = it->second;
    device.active_allocations.erase(it);

    auto event = m_streams->with_stream(device.dealloc_stream, deps, [&](auto stream) {
        KMM_CUDA_CHECK(cuMemFreeAsync(ptr, stream));
    });

    device.pending_deallocations.emplace_back(event, nbytes);
}

void* MemoryAllocatorImpl::allocate_host(size_t nbytes) {
    void* addr;

    CudaContextGuard guard {m_streams->get(DeviceId(0))};
    KMM_CUDA_CHECK(cuMemHostAlloc(
        &addr,  //
        nbytes,
        CU_MEMHOSTALLOC_PORTABLE));

    return addr;
}

void MemoryAllocatorImpl::deallocate_host(void* addr, CudaEventSet deps) {
    // All events need to be ready before memory can be deallocated.
    if (m_streams->is_ready(deps)) {
        // If the events finished, we can deallocate now immediately
        CudaContextGuard guard {m_streams->get(DeviceId(0))};
        KMM_CUDA_CHECK(cuMemFreeHost(addr));
    } else {
        // Some events are still pending, add to deletion queue to defer it to a later moment in time.
        m_deferred_deletions.push_back({std::move(deps), addr});
    }
}

void MemoryAllocatorImpl::make_progress() {
    while (!m_deferred_deletions.empty()) {
        auto& p = m_deferred_deletions.back();

        if (!m_streams->is_ready(p.dependencies)) {
            break;
        }

        CudaContextGuard guard {m_streams->get(DeviceId(0))};
        KMM_CUDA_CHECK(cuMemFreeHost(p.data));
        m_deferred_deletions.pop_back();
    }
}

// Copies smaller than this threshold are put onto a high priority stream. This can improve
// performance since small copy jobs (like copying a single number) are prioritized over large
// slow copy jobs of several gigabytes.
static constexpr size_t HIGH_PRIORITY_THRESHOLD = 1024L * 1024;

CudaEvent MemoryAllocatorImpl::copy_host_to_device(
    DeviceId device_id,
    const void* src_addr,
    CUdeviceptr dst_addr,
    size_t nbytes,
    CudaEventSet deps) {
    auto& device = m_devices.at(device_id);
    auto stream = nbytes <= HIGH_PRIORITY_THRESHOLD ? device.h2d_hi_stream : device.h2d_stream;

    return m_streams->with_stream(stream, deps, [&](auto stream) {
        KMM_CUDA_CHECK(cuMemcpyHtoDAsync(dst_addr, src_addr, nbytes, stream));
    });
}

CudaEvent MemoryAllocatorImpl::copy_device_to_host(
    DeviceId device_id,
    CUdeviceptr src_addr,
    void* dst_addr,
    size_t nbytes,
    CudaEventSet deps) {
    auto& device = m_devices.at(device_id);
    auto stream = nbytes <= HIGH_PRIORITY_THRESHOLD ? device.d2h_hi_stream : device.d2h_stream;

    return m_streams->with_stream(stream, deps, [&](auto stream) {
        KMM_CUDA_CHECK(cuMemcpyDtoHAsync(dst_addr, src_addr, nbytes, stream));
    });
}

}  // namespace kmm
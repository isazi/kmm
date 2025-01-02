#include <unordered_map>

#include "spdlog/spdlog.h"

#include "kmm/memops/gpu_fill.hpp"
#include "kmm/memops/host_fill.hpp"
#include "kmm/worker/memory_system.hpp"

namespace kmm {

struct MemorySystem::Device {
    KMM_NOT_COPYABLE(Device)

  public:
    GPUContextHandle context;
    std::unique_ptr<AsyncAllocator> allocator;

    DeviceStream h2d_stream;
    DeviceStream d2h_stream;
    DeviceStream h2d_hi_stream;  // high priority stream
    DeviceStream d2h_hi_stream;  // high priority stream

    Device(
        GPUContextHandle context,
        std::unique_ptr<AsyncAllocator> allocator,
        DeviceStreamManager& streams
    ) :
        context(context),
        allocator(std::move(allocator)),
        h2d_stream(streams.create_stream(context, false)),
        d2h_stream(streams.create_stream(context, false)),
        h2d_hi_stream(streams.create_stream(context, true)),
        d2h_hi_stream(streams.create_stream(context, true)) {}
};

MemorySystem::MemorySystem(
    std::shared_ptr<DeviceStreamManager> stream_manager,
    std::vector<GPUContextHandle> device_contexts,
    std::unique_ptr<AsyncAllocator> host_mem,
    std::vector<std::unique_ptr<AsyncAllocator>> device_mems
) :
    m_streams(stream_manager),
    m_host(std::move(host_mem))

{
    KMM_ASSERT(device_contexts.size() == device_mems.size());

    for (size_t i = 0; i < device_contexts.size(); i++) {
        m_devices[i] = std::make_unique<Device>(
            device_contexts[i],
            std::move(device_mems[i]),
            *stream_manager
        );
    }
}

MemorySystem::~MemorySystem() {}

void MemorySystem::make_progress() {
    m_host->make_progress();

    for (const auto& device : m_devices) {
        if (device == nullptr) {
            break;
        }

        device->allocator->make_progress();
    }
}

void MemorySystem::trim_host(size_t bytes_remaining) {
    m_host->trim(bytes_remaining);
}

bool MemorySystem::allocate_host(size_t nbytes, void** ptr_out, DeviceEventSet* deps_out) {
    if (!m_host->allocate_async(nbytes, ptr_out, deps_out)) {
        return false;
    }

    deps_out->remove_completed(*m_streams);
    return true;
}

void MemorySystem::deallocate_host(void* ptr, size_t nbytes, DeviceEventSet deps) {
    deps.remove_completed(*m_streams);
    return m_host->deallocate_async(ptr, nbytes, std::move(deps));
}

void MemorySystem::trim_device(size_t bytes_remaining) {
    for (const auto& device : m_devices) {
        if (device != nullptr) {
            device->allocator->trim(bytes_remaining);
        }
    }
}

bool MemorySystem::allocate_device(
    DeviceId device_id,
    size_t nbytes,
    GPUdeviceptr* ptr_out,
    DeviceEventSet* deps_out
) {
    KMM_ASSERT(m_devices[device_id]);
    auto& device = *m_devices[device_id];
    void* addr;

    GPUContextGuard guard {device.context};
    if (!device.allocator->allocate_async(nbytes, &addr, deps_out)) {
        return false;
    }

    deps_out->remove_completed(*m_streams);
    *ptr_out = (GPUdeviceptr)addr;
    return true;
}

void MemorySystem::deallocate_device(
    DeviceId device_id,
    GPUdeviceptr ptr,
    size_t nbytes,
    DeviceEventSet deps
) {
    deps.remove_completed(*m_streams);

    KMM_ASSERT(m_devices[device_id]);
    auto& device = *m_devices[device_id];

    GPUContextGuard guard {device.context};
    return device.allocator->deallocate_async((void*)ptr, nbytes, std::move(deps));
}

// Copies smaller than this threshold are put onto a high priority stream. This can improve
// performance since small copy jobs (like copying a single number) are prioritized over large
// slow copy jobs of several gigabytes.
static constexpr size_t HIGH_PRIORITY_THRESHOLD = 1024L * 1024;

DeviceEvent MemorySystem::copy_host_to_device(
    DeviceId device_id,
    const void* src_addr,
    GPUdeviceptr dst_addr,
    size_t nbytes,
    DeviceEventSet deps
) {
    KMM_ASSERT(m_devices[device_id]);
    auto& device = *m_devices[device_id];
    auto stream = nbytes <= HIGH_PRIORITY_THRESHOLD ? device.h2d_hi_stream : device.h2d_stream;

    GPUContextGuard guard {device.context};
    return m_streams->with_stream(stream, deps, [&](auto stream) {
        KMM_GPU_CHECK(gpuMemcpyHtoDAsync(dst_addr, src_addr, nbytes, stream));
    });
}

DeviceEvent MemorySystem::copy_device_to_host(
    DeviceId device_id,
    GPUdeviceptr src_addr,
    void* dst_addr,
    size_t nbytes,
    DeviceEventSet deps
) {
    KMM_ASSERT(m_devices[device_id]);
    auto& device = *m_devices[device_id];
    auto stream = nbytes <= HIGH_PRIORITY_THRESHOLD ? device.d2h_hi_stream : device.d2h_stream;

    GPUContextGuard guard {device.context};
    return m_streams->with_stream(stream, deps, [&](auto stream) {
        KMM_GPU_CHECK(gpuMemcpyDtoHAsync(dst_addr, src_addr, nbytes, stream));
    });
}

}  // namespace kmm
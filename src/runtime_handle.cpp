#include <future>

#include "kmm/cuda/device.hpp"
#include "kmm/cuda/memory.hpp"
#include "kmm/cuda/types.hpp"
#include "kmm/device.hpp"
#include "kmm/host/device.hpp"
#include "kmm/host/memory.hpp"
#include "kmm/runtime.hpp"
#include "kmm/runtime_handle.hpp"

namespace kmm {

RuntimeHandle::RuntimeHandle(std::shared_ptr<Runtime> impl) : m_impl(std::move(impl)) {
    KMM_ASSERT(m_impl);
}

MemoryId RuntimeHandle::memory_affinity_for_address(const void* address) const {
    if (address == nullptr) {
        return MemoryId(0);
    }

#ifdef KMM_USE_CUDA
    if (auto device_opt = get_cuda_device_by_address(address)) {
        for (size_t i = 0; i < m_impl->num_devices(); i++) {
            auto id = DeviceId(uint8_t(i));

            if (const auto* info = dynamic_cast<const CudaDeviceInfo*>(&m_impl->device_info(id))) {
                if (info->device() == *device_opt) {
                    return info->memory_affinity();
                }
            }
        }
    }
#endif

    return MemoryId(0);
}

EventId RuntimeHandle::submit_task(std::shared_ptr<Task> task, TaskRequirements reqs) const {
    return m_impl->submit_task(std::move(task), std::move(reqs));
}

EventId RuntimeHandle::submit_barrier() const {
    return m_impl->submit_barrier();
}

EventId RuntimeHandle::join(EventList events) const {
    if (events.size() == 1) {
        return events[0];
    }

    return m_impl->join_events(std::move(events));
}

bool RuntimeHandle::wait_until(EventId id, typename std::chrono::system_clock::time_point deadline)
    const {
    return m_impl->query_event(id, deadline);
}

bool RuntimeHandle::is_done(EventId id) const {
    return wait_until(id, std::chrono::time_point<std::chrono::system_clock> {});
}

bool RuntimeHandle::wait_for(EventId id, typename std::chrono::system_clock::duration duration)
    const {
    return wait_until(id, std::chrono::system_clock::now() + duration);
}

void RuntimeHandle::wait(EventId id) const {
    wait_until(id, std::chrono::time_point<std::chrono::system_clock>::max());
}

void RuntimeHandle::synchronize() const {
    m_impl->query_event(m_impl->submit_barrier());
}

RuntimeHandle build_runtime() {
    auto host_device = std::make_shared<ParallelDeviceHandle>();

    std::vector<std::shared_ptr<DeviceHandle>> handles = {host_device};
    std::unique_ptr<Memory> memory;

#ifdef KMM_USE_CUDA
    auto cuda_devices = get_cuda_devices();

    if (!cuda_devices.empty()) {
        auto contexts = std::vector<CudaContextHandle> {};
        uint8_t memory_id = 1;

        for (auto cuda_device : cuda_devices) {
            auto context = CudaContextHandle::create_context_for_device(cuda_device);
            contexts.push_back(context);

            handles.push_back(std::make_shared<CudaDeviceHandle>(context, MemoryId(memory_id)));
            memory_id++;
        }

        memory = std::make_unique<CudaMemory>(host_device, contexts);
    } else
#endif  // KMM_USE_CUDA
    {
        memory = std::make_unique<HostMemory>(host_device);
    }

    return std::make_shared<Runtime>(std::move(handles), std::move(memory));
}

}  // namespace kmm
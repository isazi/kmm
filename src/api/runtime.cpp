#include "kmm/api/runtime.hpp"
#include "kmm/internals/memory_manager.hpp"
#include "kmm/internals/scheduler.hpp"

namespace kmm {

EventId Runtime::join(EventList events) const {
    return m_impl->join_events(events);
}

bool Runtime::wait_until(EventId id, std::chrono::system_clock::time_point deadline) const {
    return m_impl->query_event(id, deadline);
}

bool Runtime::is_done(EventId id) const {
    return wait_until(id, std::chrono::system_clock::time_point::min());
}

void Runtime::wait(EventId id) const {
    wait_until(id, std::chrono::system_clock::time_point::max());
}

bool Runtime::wait_for(EventId id, std::chrono::system_clock::duration duration) const {
    auto now = std::chrono::system_clock::now();
    auto max_duration = std::chrono::system_clock::time_point::max() - now;

    return wait_until(id, now + std::min(max_duration, duration));
}

EventId Runtime::submit_barrier() const {
    return m_impl->insert_barrier();
}

void Runtime::synchronize() const {
    wait(submit_barrier());
}

const std::vector<DeviceInfo>& Runtime::devices() const {
    return m_impl->devices();
}

Runtime make_runtime() {
    auto contexts = std::vector<CudaContextHandle>();
    auto devices = std::vector<MemoryDeviceInfo>();

    for (const auto& device : get_cuda_devices()) {
        auto context = CudaContextHandle::create_context_for_device(device);

        contexts.push_back(context);
        devices.push_back(MemoryDeviceInfo {
            .context = context,
        });
    }

    auto stream_manager = std::make_shared<CudaStreamManager>(contexts, 4);
    auto memory_manager = std::make_shared<MemoryManager>(stream_manager, devices);
    auto scheduler = std::make_shared<Scheduler>(memory_manager, stream_manager);

    return std::make_shared<RuntimeImpl>(scheduler, memory_manager);
}

}  // namespace kmm
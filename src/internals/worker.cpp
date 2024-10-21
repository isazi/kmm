#include <thread>

#include "spdlog/spdlog.h"

#include "kmm/internals/allocator/cuda.hpp"
#include "kmm/internals/allocator/system.hpp"
#include "kmm/internals/worker.hpp"

namespace kmm {

bool Worker::query_event(EventId event_id, std::chrono::system_clock::time_point deadline) {
    static constexpr auto TIMEOUT = std::chrono::microseconds {100};

    std::unique_lock guard {m_mutex};
    flush_events_impl();

    if (m_executor.is_completed(event_id)) {
        return true;
    }

    auto next_update = std::chrono::system_clock::now();

    while (true) {
        make_progress_impl();

        if (m_executor.is_completed(event_id)) {
            return true;
        }

        auto now = std::chrono::system_clock::now();
        next_update += TIMEOUT;

        if (next_update < now) {
            next_update = now;
        }

        if (next_update > deadline) {
            return false;
        }

        guard.unlock();
        std::this_thread::sleep_until(next_update);
        guard.lock();
    }
}

bool Worker::is_idle() {
    std::lock_guard guard {m_mutex};
    flush_events_impl();
    is_idle_impl();
}

void Worker::make_progress() {
    std::lock_guard guard {m_mutex};
    flush_events_impl();
    make_progress_impl();
}

void Worker::shutdown() {
    std::unique_lock guard {m_mutex};
    if (m_has_shutdown) {
        return;
    }

    m_has_shutdown = true;

    m_graph.shutdown();
    flush_events_impl();

    while (!is_idle_impl()) {
        make_progress_impl();

        guard.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds {10});
        guard.lock();
    }

    m_stream_manager->wait_until_idle();
}

void Worker::flush_events_impl() {
    // Flush all events from the DAG builder to the scheduler
    for (auto event : m_graph.flush()) {
        m_executor
            .submit_command(event.id, std::move(event.command), std::move(event.dependencies));
    }
}

void Worker::make_progress_impl() {
    m_stream_manager->make_progress();
    m_memory_manager->make_progress();
    m_executor.make_progress();
}

bool Worker::is_idle_impl() {
    return m_stream_manager->is_idle() && m_memory_manager->is_idle(*m_stream_manager)
        && m_executor.is_idle();
}

Worker::~Worker() {
    shutdown();
}

SystemInfo make_system_info(const std::vector<CudaContextHandle>& contexts) {
    spdlog::info("detected {} CUDA device(s):", contexts.size());
    std::vector<CudaDeviceInfo> device_infos;

    for (size_t i = 0; i < contexts.size(); i++) {
        auto info = CudaDeviceInfo(DeviceId(i), contexts[i]);

        spdlog::info(" - {} ({:.2} GB)", info.name(), info.total_memory_size() / 1e9);
        device_infos.push_back(info);
    }

    return device_infos;
}

Worker::Worker(
    std::vector<CudaContextHandle> contexts,
    std::shared_ptr<CudaStreamManager> stream_manager,
    std::shared_ptr<MemoryManager> memory_manager
) :
    m_info(make_system_info(contexts)),
    m_executor(contexts, stream_manager, memory_manager),
    m_stream_manager(stream_manager),
    m_memory_manager(memory_manager) {}

std::shared_ptr<Worker> make_worker() {
    std::unique_ptr<MemoryAllocator> host_mem;
    std::vector<std::unique_ptr<MemoryAllocator>> device_mems;

    auto stream_manager = std::make_shared<CudaStreamManager>();
    auto contexts = std::vector<CudaContextHandle>();
    auto devices = get_cuda_devices();

    if (!devices.empty()) {
        for (size_t i = 0; i < contexts.size(); i++) {
            auto context = CudaContextHandle::retain_primary_context_for_device(devices[i]);
            device_mems.push_back(std::make_unique<DevicePoolAllocator>(context, stream_manager));
            contexts.push_back(context);
        }

        host_mem = std::make_unique<PinnedMemoryAllocator>(contexts.at(0), stream_manager);
    } else {
        host_mem = std::make_unique<SystemAllocator>(stream_manager);
    }

    auto memory_system = std::make_unique<MemorySystem>(
        stream_manager,
        contexts,
        std::move(host_mem),
        std::move(device_mems)
    );

    auto memory_manager = std::make_shared<MemoryManager>(std::move(memory_system));
    return std::make_shared<Worker>(contexts, stream_manager, memory_manager);
}
}  // namespace kmm
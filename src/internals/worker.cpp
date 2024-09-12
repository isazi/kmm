#include <thread>

#include "spdlog/spdlog.h"

#include "kmm/internals/worker.hpp"

namespace kmm {

bool Worker::query_event(EventId event_id, std::chrono::system_clock::time_point deadline) {
    static constexpr auto TIMEOUT = std::chrono::microseconds {100};

    std::unique_lock guard {m_mutex};
    flush_events_impl();

    if (m_scheduler->is_completed(event_id)) {
        return true;
    }

    auto next_update = std::chrono::system_clock::now();

    while (true) {
        make_progress_impl();

        if (m_scheduler->is_completed(event_id)) {
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

    m_graph->shutdown();
    flush_events_impl();

    while (!m_scheduler->is_idle() || !m_executor->is_idle() || !m_memory->is_idle(*m_streams)) {
        make_progress_impl();

        guard.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds {10});
        guard.lock();
    }

    m_streams->wait_until_idle();
}

struct BufferGuardTracker {
    KMM_NOT_COPYABLE_OR_MOVABLE(BufferGuardTracker)

  public:
    BufferGuardTracker(
        std::shared_ptr<Worker> worker,
        std::shared_ptr<MemoryManager::Request> request) :
        m_worker(std::move(worker)),
        m_request(std::move(request)) {
        KMM_ASSERT(m_worker != nullptr && m_request != nullptr);
    }

    ~BufferGuardTracker() {
        std::unique_lock guard {m_worker->m_mutex};
        m_worker->m_memory->release_request(m_request);
    }

    std::shared_ptr<Worker> m_worker;
    std::shared_ptr<MemoryManager::Request> m_request;
};

BufferGuard Worker::access_buffer(BufferId buffer_id, MemoryId memory_id, AccessMode mode) {
    // We must declare the tracker before the `unique_lock`. This way, if an exception is thrown, the lock is released
    // before the tracker is destroyed. This is necessary since the tracker also obtains the worker lock.
    std::shared_ptr<BufferGuardTracker> tracker;

    std::unique_lock guard {m_mutex};
    flush_events_impl();
    make_progress_impl();

    // Create the request and immediately put into the tracker. This ensures that the request is
    // always released in case an exception is thrown while polling.
    auto buffer = m_buffers->get(buffer_id);
    auto req = m_memory->create_request(buffer, memory_id, mode, m_root_transaction);
    tracker = std::make_shared<BufferGuardTracker>(shared_from_this(), req);

    CudaEventSet deps;

    // Poll until the request is ready.
    while (!m_memory->poll_request(*req, deps) || !m_streams->is_ready(deps)) {
        make_progress_impl();

        guard.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds {10});
        guard.lock();
    }

    return {m_memory->get_accessor(*req), std::move(tracker)};
}

void Worker::flush_events_impl() {
    // Flush all events from the DAG builder to the scheduler
    for (auto event : m_graph->flush()) {
        m_scheduler->insert_event(
            event.id,
            std::move(event.command),
            std::move(event.dependencies));
    }
}

void Worker::make_progress_impl() {
    flush_events_impl();
    bool update_happened = true;

    while (update_happened) {
        update_happened = false;
        m_streams->make_progress();
        m_memory->make_progress();
        m_executor->make_progress();

        while (auto node = m_scheduler->pop_ready()) {
            update_happened = true;
            execute_command(*node);
        }
    }
}

void Worker::execute_command(std::shared_ptr<TaskNode> node) {
    const Command& command = node->get_command();

    if (const auto* e = std::get_if<CommandEmpty>(&command)) {
        m_scheduler->set_complete(node);

    } else if (const auto* e = std::get_if<CommandBufferCreate>(&command)) {
        spdlog::debug("create buffer {} (size={})", e->id, e->layout.size_in_bytes);
        auto buffer = m_memory->create_buffer(e->layout);
        m_buffers->add(e->id, buffer);
        m_scheduler->set_complete(node);

    } else if (const auto* e = std::get_if<CommandBufferDelete>(&command)) {
        spdlog::debug("delete buffer {}", e->id);
        auto buffer = m_buffers->remove(e->id);
        m_memory->delete_buffer(buffer);
        m_scheduler->set_complete(node);

    } else if (const auto* e = std::get_if<CommandPrefetch>(&command)) {
        m_executor->submit_prefetch(node, e->buffer_id, e->memory_id);

    } else if (const auto* e = std::get_if<CommandCopy>(&command)) {
        m_executor->submit_copy(
            node,
            e->src_buffer,
            e->src_memory,
            e->dst_buffer,
            e->dst_memory,
            e->spec);

    } else if (const auto* e = std::get_if<CommandExecute>(&command)) {
        m_executor->submit_task(node, e->processor_id, e->task, e->buffers);

    } else {
        KMM_PANIC("invalid command");
    }
}

Worker::Worker(std::vector<CudaContextHandle> contexts) {
    m_streams = std::make_shared<CudaStreamManager>(contexts);
    m_scheduler = std::make_shared<Scheduler>(contexts.size());
    m_graph = std::make_shared<TaskGraph>();
    m_buffers = std::make_shared<BufferManager>();

    std::vector<CudaDeviceInfo> device_infos;
    std::vector<MemoryDeviceInfo> device_mems;

    for (size_t i = 0; i < contexts.size(); i++) {
        auto device_id = DeviceId(i);
        auto context = contexts[i];

        device_infos.push_back(CudaDeviceInfo(device_id, context));
        device_mems.push_back(MemoryDeviceInfo {
            .num_bytes_limit = 5'000'000'000,
        });
    }

    m_info = SystemInfo(device_infos);

    m_memory = std::make_shared<MemoryManager>(
        std::make_unique<MemoryAllocatorImpl>(m_streams, device_mems));
    m_root_transaction = m_memory->create_transaction();

    m_executor =
        std::make_shared<Executor>(contexts.size(), m_streams, m_buffers, m_memory, m_scheduler);
}

Worker::~Worker() {
    shutdown();
}

std::shared_ptr<Worker> make_worker() {
    std::vector<CUdevice> devices = get_cuda_devices();
    std::vector<CudaContextHandle> contexts;

    for (auto device : devices) {
        contexts.push_back(CudaContextHandle::retain_primary_context_for_device(device));
    }

    return std::make_shared<Worker>(std::move(contexts));
}
}  // namespace kmm
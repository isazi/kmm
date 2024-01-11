#include "kmm/cuda/copy_engine.hpp"

#ifdef USE_CUDA

namespace kmm {

CUresult CudaCopyEngine::CopyJob::execute(CUstream stream) const {
    return cuMemcpyAsync(
        reinterpret_cast<CUdeviceptr>(dst_ptr),
        reinterpret_cast<CUdeviceptr>(src_ptr),
        num_bytes,
        stream);
}

struct CudaCopyEngine::TransferQueue {
    void initialize(int priority) {
        unsigned int flags = CU_STREAM_NON_BLOCKING;
        KMM_CUDA_CHECK(cuStreamCreateWithPriority(&m_stream, flags, priority));

        for (size_t i = 0; i < MAX_CONCURRENT_EVENTS; i++) {
            KMM_CUDA_CHECK(cuEventCreate(&m_events[i], 0));
        }
    }

    void destroy() {
        if (m_stream != nullptr) {
            KMM_CUDA_CHECK(cuStreamDestroy(m_stream));
            m_stream = nullptr;
        }

        for (auto& event : m_events) {
            if (event != nullptr) {
                KMM_CUDA_CHECK(cuEventDestroy(event));
                event = nullptr;
            }
        }
    }

    bool make_progress() {
        while (m_num_running > 0) {
            size_t slot = m_head;
            auto result = cuEventQuery(m_events[slot]);

            if (result == CUDA_ERROR_NOT_READY) {
                break;
            }

            KMM_ASSERT(cuEventSynchronize(m_events[slot]) == CUDA_SUCCESS);

            auto completion = std::move(m_running_jobs[slot]);
            m_head = (slot + 1) % m_events.size();
            m_num_running--;

            if (result == CUDA_SUCCESS) {
                completion.complete_ok();
            } else {
                completion.complete_error(CudaException("operation failed", result));
            }
        }

        while (m_num_running < m_events.size() && !m_waiting_jobs.empty()) {
            auto job = std::move(m_waiting_jobs.front());
            m_waiting_jobs.pop_front();

            execute_job(std::move(job));
        }

        return m_num_running > 0;
    }

    void execute_job(CopyJob&& job) {
        size_t slot = (m_head + m_num_running) % m_events.size();
        KMM_ASSERT(!m_running_jobs[slot]);

        try {
            KMM_CUDA_CHECK(cuEventSynchronize(m_events[slot]));

            auto result = job.execute(m_stream);

            if (result != CUDA_SUCCESS) {
                throw CudaException("copy operation failed", result);
            }

            result = cuEventRecord(m_events[slot], m_stream);

            if (result != CUDA_SUCCESS) {
                KMM_ASSERT(cuStreamSynchronize(m_stream) == CUDA_SUCCESS);
                throw CudaException("`cuEventRecord` failed", result);
            }

            m_running_jobs[slot] = std::move(job.completion);
            m_num_running++;
        } catch (...) {
            job.completion.complete(ErrorPtr::from_current_exception());
        }
    }

    void submit_job(CopyJob&& job) {
        if (m_num_running < m_events.size()) {
            execute_job(std::move(job));
        } else {
            m_waiting_jobs.emplace_back(std::move(job));
        }
    }

  private:
    static constexpr size_t MAX_CONCURRENT_EVENTS = 5;

    CUstream m_stream = nullptr;
    std::array<CUevent, MAX_CONCURRENT_EVENTS> m_events = {nullptr};
    std::array<Completion, MAX_CONCURRENT_EVENTS> m_running_jobs;
    size_t m_head = 0;
    size_t m_num_running = 0;
    std::deque<CopyJob> m_waiting_jobs;
};

struct CudaCopyEngine::Device {
    static constexpr size_t HIGH_PRIORITY_MAX_BYTES = 1024;

    CudaContextHandle context;
    TransferQueue h2d;
    TransferQueue d2h;
    TransferQueue h2d_high_prio;
    TransferQueue d2h_high_prio;

    Device(const CudaContextHandle& context) : context(context) {
        CudaContextGuard guard {context};

        int least_priority = 0;
        int greatest_priority = 0;
        KMM_CUDA_CHECK(cuCtxGetStreamPriorityRange(&least_priority, &greatest_priority));

        h2d.initialize(least_priority);
        d2h.initialize(least_priority);
        h2d_high_prio.initialize(greatest_priority);
        d2h_high_prio.initialize(greatest_priority);
    }

    ~Device() {
        CudaContextGuard guard {context};
        h2d.destroy();
        d2h.destroy();
        h2d_high_prio.destroy();
        d2h_high_prio.destroy();
    }

    void submit_job(CopyJob&& job) {
        CudaContextGuard guard {context};
        bool is_high_priority = job.num_bytes <= HIGH_PRIORITY_MAX_BYTES;

        if (job.src_is_host) {
            if (is_high_priority) {
                h2d_high_prio.submit_job(std::move(job));
            } else {
                h2d.submit_job(std::move(job));
            }
        } else {
            if (is_high_priority) {
                d2h_high_prio.submit_job(std::move(job));
            } else {
                d2h.submit_job(std::move(job));
            }
        }
    }

    bool make_progress() {
        CudaContextGuard guard {context};
        bool is_active = false;
        is_active |= h2d_high_prio.make_progress();
        is_active |= d2h_high_prio.make_progress();
        is_active |= h2d.make_progress();
        is_active |= d2h.make_progress();
        return is_active;
    }
};

CudaCopyEngine::CudaCopyEngine(std::vector<CudaContextHandle> contexts) {
    for (const auto& context : contexts) {
        m_devices.emplace_back(std::make_unique<Device>(context));
    }
}

CudaCopyEngine::~CudaCopyEngine() = default;

void CudaCopyEngine::copy_host_to_device_async(
    size_t dst_device,
    const void* src_data,
    void* dst_data,
    size_t num_bytes,
    Completion completion) {
    submit_job(CopyJob {
        .device_index = dst_device,
        .src_is_host = true,
        .src_ptr = src_data,
        .dst_ptr = dst_data,
        .num_bytes = num_bytes,
        .completion = std::move(completion),
    });
}

void CudaCopyEngine::copy_device_to_host_async(
    size_t src_device,
    const void* src_data,
    void* dst_data,
    size_t num_bytes,
    Completion completion) {
    submit_job(CopyJob {
        .device_index = src_device,
        .src_is_host = false,
        .src_ptr = src_data,
        .dst_ptr = dst_data,
        .num_bytes = num_bytes,
        .completion = std::move(completion),
    });
}

void CudaCopyEngine::copy_device_to_device_async(
    size_t src_device,
    size_t dst_device,
    const void* src_data,
    void* dst_data,
    size_t num_bytes,
    Completion completion) {
    submit_job(CopyJob {
        .device_index = src_device,
        .src_is_host = false,
        .src_ptr = src_data,
        .dst_ptr = dst_data,
        .num_bytes = num_bytes,
        .completion = std::move(completion),
    });
}

void CudaCopyEngine::fill_device_async(
    size_t dst_device,
    void* dst_data,
    size_t num_bytes,
    std::vector<uint8_t> fill_pattern,
    Completion completion) {
    completion.complete_error("fill is not supported");
}

void CudaCopyEngine::run_forever() {
    auto next_update = std::chrono::system_clock::now();
    bool copies_in_progress = false;
    bool shutdown_requested = false;

    while (copies_in_progress || !shutdown_requested) {
        std::optional<CopyJob> new_job = wait_for_new_job(next_update, shutdown_requested);

        auto before_update = std::chrono::system_clock::now();
        copies_in_progress = submit_job_and_make_progress(std::move(new_job));

        // If there are any copies in progress, we sleep for just 10 microseconds
        // If there are no copies in progress, we wait for much longer
        next_update = before_update
            + (copies_in_progress ? std::chrono::microseconds(10) : std::chrono::milliseconds(100));
    }
}

bool CudaCopyEngine::submit_job_and_make_progress(std::optional<CopyJob> new_job) {
    std::lock_guard guard {m_mutex};

    if (new_job) {
        m_devices[new_job->device_index]->submit_job(std::move(*new_job));
    }

    bool copies_in_progress = false;

    for (const auto& device : m_devices) {
        copies_in_progress |= device->make_progress();
    }

    return copies_in_progress;
}

std::optional<CudaCopyEngine::CopyJob> CudaCopyEngine::wait_for_new_job(
    std::chrono::system_clock::time_point deadline,
    bool& shutdown_requested) {
    std::unique_lock guard {m_queue_mutex};
    shutdown_requested = m_queue_closed;

    if (m_queue.empty()) {
        m_queue_cond.wait_until(guard, deadline);
        return std::nullopt;
    }

    auto new_job = std::move(this->m_queue.front());
    m_queue.pop_front();
    return new_job;
}

void CudaCopyEngine::submit_job(CopyJob&& job) {
    std::unique_lock queue_guard {m_queue_mutex};
    if (m_queue_closed) {
        queue_guard.unlock();
        job.completion.complete_error("copy failed since system is shutting down");
        return;
    }

    m_queue_cond.notify_all();

    if (std::unique_lock g = {m_mutex, std::try_to_lock}) {
        queue_guard.unlock();
        m_devices[job.device_index]->submit_job(std::move(job));
        return;
    }

    m_queue.push_back(std::move(job));
}

void CudaCopyEngine::shutdown() {
    std::lock_guard guard {m_queue_mutex};
    m_queue_closed = true;
    m_queue_cond.notify_all();
}

}  // namespace kmm

#endif  // USE_CUDA
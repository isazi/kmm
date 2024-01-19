
#include "kmm/cuda/device.hpp"

#ifdef KMM_USE_CUDA

namespace kmm {

CudaDevice::CudaDevice(CudaContextHandle context, MemoryId affinity_id) :
    CudaDeviceInfo(context, affinity_id),
    m_context(context) {
    CudaContextGuard guard {context};

    unsigned int flags = 0;
    KMM_CUDA_CHECK(cuStreamCreate(&m_stream, flags));

    flags = CU_EVENT_DEFAULT;
    KMM_CUDA_CHECK(cuEventCreate(&m_event, flags));
}

CudaDevice::~CudaDevice() noexcept {
    KMM_CUDA_CHECK(cuStreamDestroy(m_stream));
    KMM_CUDA_CHECK(cuEventDestroy(m_event));

    if (m_cublas_handle != nullptr) {
        cublasDestroy(m_cublas_handle);
    }
}

void CudaDevice::synchronize() const {
    CudaContextGuard guard {m_context};
    KMM_CUDA_CHECK(cuStreamSynchronize(m_stream));
    KMM_CUDA_CHECK(cuStreamSynchronize(CUDA_DEFAULT_STREAM));
}

cublasHandle_t CudaDevice::cublas() {
    if (m_cublas_handle == nullptr) {
        KMM_CUDA_CHECK(cublasCreate(&m_cublas_handle));
        KMM_CUDA_CHECK(cublasSetStream(m_cublas_handle, m_stream));
    }

    return m_cublas_handle;
}

static bool is_fill_pattern_repetitive(
    size_t k,
    const void* fill_pattern,
    size_t fill_pattern_size) {
    if (fill_pattern_size < k || fill_pattern_size % k != 0) {
        return false;
    }

    for (size_t i = 1; i < fill_pattern_size / k; i++) {
        for (size_t j = 0; j < k; j++) {
            if (static_cast<const char*>(fill_pattern)[i * k + j]
                != static_cast<const char*>(fill_pattern)[j]) {
                return false;
            }
        }
    }

    return true;
}

void CudaDevice::fill_raw(
    void* dest_buffer,
    size_t nbytes,
    const void* fill_pattern,
    size_t fill_pattern_size) const {
    if (is_fill_pattern_repetitive(1, fill_pattern, fill_pattern_size)) {
        uint8_t fill_value;
        ::memcpy(&fill_value, fill_pattern, sizeof(uint8_t));

        KMM_CUDA_CHECK(cuMemsetD8Async(CUdeviceptr(dest_buffer), fill_value, nbytes, m_stream));

    } else if (is_fill_pattern_repetitive(2, fill_pattern, fill_pattern_size)) {
        uint16_t fill_value;
        ::memcpy(&fill_value, fill_pattern, sizeof(uint16_t));

        KMM_CUDA_CHECK(cuMemsetD16Async(
            CUdeviceptr(dest_buffer),
            fill_value,
            nbytes / sizeof(uint16_t),
            m_stream));

    } else if (is_fill_pattern_repetitive(4, fill_pattern, fill_pattern_size)) {
        uint32_t fill_value;
        ::memcpy(&fill_value, fill_pattern, sizeof(uint32_t));

        KMM_CUDA_CHECK(cuMemsetD32Async(
            CUdeviceptr(dest_buffer),
            fill_value,
            nbytes / sizeof(uint32_t),
            m_stream));

    } else {
        throw CudaException(fmt::format(
            "could not fill buffer, value is {} bit, but only 8, 16, or 32 bit is supported",
            fill_pattern_size * 8));
    }
}

void CudaDevice::copy_raw(const void* source_buffer, void* dest_buffer, size_t nbytes) const {
    KMM_CUDA_CHECK(
        cuMemcpyDtoDAsync(CUdeviceptr(dest_buffer), CUdeviceptr(source_buffer), nbytes, m_stream));
}

class CudaDeviceThread {
  public:
    CudaDeviceThread(
        std::shared_ptr<WorkQueue<CudaDeviceHandle::Job>> queue,
        std::vector<std::unique_ptr<CudaDevice>> streams);
    ~CudaDeviceThread();

    void run_forever();

  private:
    PollResult poll_stream(size_t slot);
    void submit_job(size_t slot, std::unique_ptr<CudaDeviceHandle::Job> job);

    std::shared_ptr<WorkQueue<CudaDeviceHandle::Job>> m_queue;
    std::vector<std::unique_ptr<CudaDevice>> m_streams;
    std::vector<Completion> m_running_jobs;
    std::vector<CUevent> m_events;
};

CudaDeviceThread::CudaDeviceThread(
    std::shared_ptr<WorkQueue<CudaDeviceHandle::Job>> queue,
    std::vector<std::unique_ptr<CudaDevice>> streams) :
    m_queue(std::move(queue)),
    m_streams(std::move(streams)) {
    KMM_ASSERT(m_streams.empty() == false);

    m_running_jobs.resize(m_streams.size());

    for (const auto& stream : m_streams) {
        CudaContextGuard guard {stream->context_handle()};

        CUevent event;
        KMM_CUDA_CHECK(cuEventCreate(&event, 0));
        m_events.push_back(event);
    }
}

CudaDeviceThread::~CudaDeviceThread() {
    for (const auto& event : m_events) {
        KMM_CUDA_CHECK(cuEventDestroy(event));
    }

    m_events.clear();
    m_queue->shutdown();
}

void CudaDeviceThread::run_forever() {
    CudaContextGuard guard {m_streams[0]->context_handle()};

    bool shutdown = false;

    while (!shutdown) {
        size_t num_free_slots = 0;
        size_t first_free_slot = 0;

        auto next_update = std::chrono::system_clock::now() + std::chrono::microseconds(50);

        // First, poll each stream
        for (size_t i = 0; i < m_streams.size(); i++) {
            if (poll_stream(i) == PollResult::Ready) {
                if (num_free_slots == 0) {
                    first_free_slot = i;
                }

                num_free_slots++;
            }
        }

        // Next, there are four possible options:
        // 1. there are no free slots, simply sleep until the next update
        if (num_free_slots == 0) {
            std::this_thread::sleep_until(next_update);
        }

        // 2. there is a free slot, sleep until the next update or until a new job arrives.
        else if (num_free_slots < m_streams.size()) {
            if (auto job_opt = m_queue->pop_wait_until(next_update)) {
                submit_job(first_free_slot, std::move(*job_opt));
            }
        }

        // 3. all slots are free, sleep until the next job arrives. If `pop` is empty, then
        //    it means that the queue has been closed and this thread should shut down.
        else {
            if (auto job_opt = m_queue->pop()) {
                submit_job(0, std::move(*job_opt));
            } else {
                shutdown = true;
            }
        }
    }
}

PollResult CudaDeviceThread::poll_stream(size_t slot) {
    // no job on this slot, return `Ready`
    if (!m_running_jobs[slot]) {
        return PollResult::Ready;
    }

    // Query the stream at this slot. If it is not ready, we return `Pending`.
    auto status = cuStreamQuery(m_streams[slot]->stream());
    if (status == CUDA_ERROR_NOT_READY) {
        return PollResult::Pending;
    }

    // Make sure that the stream is really done. It could be that the error returned by
    // `cuStreamQuery` was some other asynchronous error. We need to be sure the work on this
    // stream has really finished before we fire the completion.
    KMM_CUDA_CHECK(cuStreamSynchronize(m_streams[slot]->stream()));

    auto completion = std::move(m_running_jobs[slot]);
    if (status == CUDA_SUCCESS) {
        completion.complete_ok();
    } else {
        completion.complete_error(CudaDriverException("execution failed", status));
    }

    return PollResult::Ready;
}

void CudaDeviceThread::submit_job(size_t slot, std::unique_ptr<CudaDeviceHandle::Job> job) {
    KMM_ASSERT(!m_running_jobs[slot]);

    try {
        job->task->execute(*m_streams[slot], job->context);

        KMM_CUDA_CHECK(cuEventRecord(m_events[slot], CUDA_DEFAULT_STREAM));
        KMM_CUDA_CHECK(cuStreamWaitEvent(m_streams[slot]->stream(), m_events[slot], 0));

        m_running_jobs[slot] = std::move(job->completion);
    } catch (...) {
        KMM_ASSERT(cuStreamSynchronize(m_streams[slot]->stream()) == CUDA_SUCCESS);

        job->completion.complete_error(ErrorPtr::from_current_exception());
    }
}

CudaDeviceHandle::Job::Job(std::shared_ptr<Task> task, TaskContext context, Completion completion) :
    task(std::move(task)),
    context(std::move(context)),
    completion(std::move(completion)) {

    };

CudaDeviceHandle::CudaDeviceHandle(
    CudaContextHandle context,
    MemoryId affinity_id,
    size_t num_streams) :
    m_info(context, affinity_id),
    m_queue(std::make_shared<WorkQueue<Job>>()) {
    std::vector<std::unique_ptr<CudaDevice>> streams;

    for (size_t i = 0; i < num_streams; i++) {
        streams.push_back(std::make_unique<CudaDevice>(context, affinity_id));
    }

    auto state = std::make_unique<CudaDeviceThread>(m_queue, std::move(streams));
    m_thread = std::thread([state = std::move(state)] { state->run_forever(); });
}

CudaDeviceHandle::~CudaDeviceHandle() noexcept {
    m_queue->shutdown();
    m_thread.join();
}

std::unique_ptr<DeviceInfo> CudaDeviceHandle::info() const {
    return std::make_unique<CudaDeviceInfo>(m_info);
}
void CudaDeviceHandle::submit(
    std::shared_ptr<Task> task,
    TaskContext context,
    Completion completion) const {
    auto job = std::make_unique<Job>(std::move(task), std::move(context), std::move(completion));
    m_queue->push(std::move(job));
}
}  // namespace kmm

#endif  // KMM_USE_CUDA
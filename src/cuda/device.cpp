#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include "kmm/cuda/device.hpp"

#ifdef KMM_USE_CUDA

namespace kmm {

CudaDeviceInfo::CudaDeviceInfo(CudaContextHandle context, MemoryId affinity_id) :
    m_affinity_id(affinity_id) {
    CudaContextGuard guard {context};

    KMM_CUDA_CHECK(cuCtxGetDevice(&m_device_id));

    char name[1024];
    KMM_CUDA_CHECK(cuDeviceGetName(name, 1024, m_device_id));
    m_name = std::string(name);

    for (size_t i = 0; i < NUM_ATTRIBUTES; i++) {
        auto attr = CUdevice_attribute(i);
        KMM_CUDA_CHECK(cuDeviceGetAttribute(&m_attributes[i], attr, m_device_id));
    }

    size_t ignore_free_memory;
    KMM_CUDA_CHECK(cuMemGetInfo(&ignore_free_memory, &m_memory_capacity));
}

dim3 CudaDeviceInfo::max_block_dim() const {
    return dim3(
        attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X),
        attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y),
        attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z));
}

dim3 CudaDeviceInfo::max_grid_dim() const {
    return dim3(
        attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X),
        attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y),
        attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z));
}

int CudaDeviceInfo::compute_capability() const {
    return attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR) * 10
        + attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
}

int CudaDeviceInfo::max_threads_per_block() const {
    return attribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
}

size_t CudaDeviceInfo::total_memory_size() const {
    return m_memory_capacity;
}

int CudaDeviceInfo::attribute(CUdevice_attribute attrib) const {
    if (attrib < NUM_ATTRIBUTES) {
        return m_attributes[attrib];
    }

    throw std::runtime_error("unsupported attribute requested");
}

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
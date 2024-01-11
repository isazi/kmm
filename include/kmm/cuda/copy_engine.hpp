#pragma once

#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <vector>

#include "kmm/completion.hpp"
#include "kmm/cuda/types.hpp"

#ifdef USE_CUDA

namespace kmm {

class CudaCopyEngine {
    struct TransferQueue;
    struct Device;
    struct CopyJob {
        size_t device_index;
        bool src_is_host = false;
        const void* src_ptr;
        void* dst_ptr;
        size_t num_bytes;
        Completion completion;

        CUresult execute(CUstream stream) const;
    };

  public:
    explicit CudaCopyEngine(std::vector<CudaContextHandle> contexts);
    ~CudaCopyEngine();

    void copy_host_to_device_async(
        size_t dst_device,
        const void* src_data,
        void* dst_data,
        size_t num_bytes,
        Completion completion);

    void copy_device_to_host_async(
        size_t src_device,
        const void* src_data,
        void* dst_data,
        size_t num_bytes,
        Completion completion);

    void copy_device_to_device_async(
        size_t src_device,
        size_t dst_device,
        const void* src_data,
        void* dst_data,
        size_t num_bytes,
        Completion completion);

    void fill_device_async(
        size_t dst_device,
        void* dst_data,
        size_t num_bytes,
        std::vector<uint8_t> fill_pattern,
        Completion completion);

    void run_forever();
    void shutdown();

  private:
    void submit_job(CopyJob&& job);
    std::optional<CopyJob> wait_for_new_job(
        std::chrono::system_clock::time_point deadline,
        bool& shutdown_requested);
    bool submit_job_and_make_progress(std::optional<CopyJob> new_job);

    std::mutex m_mutex;
    std::vector<std::unique_ptr<Device>> m_devices;

    std::mutex m_queue_mutex;
    bool m_queue_closed = false;
    std::condition_variable m_queue_cond;
    std::deque<CopyJob> m_queue;
};
}  // namespace kmm

#endif  // USE_CUDA
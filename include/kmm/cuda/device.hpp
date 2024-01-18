#pragma once

#include <array>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#ifdef KMM_USE_CUDA
    #include <cuda.h>
    #include <cuda_runtime_api.h>
#endif

#include "kmm/cuda/types.hpp"
#include "kmm/device.hpp"
#include "kmm/host/work_queue.hpp"
#include "kmm/identifiers.hpp"

#ifdef KMM_USE_CUDA

namespace kmm {

/**
 * Stores the information on a CUDA device.
 */
class CudaDeviceInfo: public DeviceInfo {
  public:
    static constexpr size_t NUM_ATTRIBUTES = CU_DEVICE_ATTRIBUTE_MAX;

    CudaDeviceInfo(CudaContextHandle context, MemoryId affinity_id);

    /**
     * Returns the name of the CUDA device as provided by `cuDeviceGetName`.
     */
    std::string name() const override {
        return m_name;
    }

    /**
     * Returns which memory this device has affinity to.
     */
    MemoryId memory_affinity() const override {
        return m_affinity_id;
    }

    /**
     * Return this device as a `CUdevice`.
     */
    CUdevice device() const {
        return m_device_id;
    }

    /**
     * Returns the maximum block size supported by this device.
     */
    dim3 max_block_dim() const;

    /**
     * Returns the maximum grid size supported by this device.
     */
    dim3 max_grid_dim() const;

    /**
     * Returns the compute capability of this device as integer `MAJOR * 10 + MINOR` (For example,
     * `86` means capability 8.6)
     */
    int compute_capability() const;

    /**
     * Returns the maximum number of threads per block supported by this device.
     */
    int max_threads_per_block() const;

    /**
     * Returns the total memory size of this device.
     */
    size_t total_memory_size() const;

    /**
     * Returns the value of the provided attribute.
     */
    int attribute(CUdevice_attribute attrib) const;

  private:
    std::string m_name;
    CUdevice m_device_id;
    size_t m_memory_capacity;
    MemoryId m_affinity_id;
    std::array<int, NUM_ATTRIBUTES> m_attributes;
};

/**
 * Contains the state of a CUDA device, such as the CUDA context and the CUDA stream.
 */
class CudaDevice final: public CudaDeviceInfo, public Device {
    KMM_NOT_COPYABLE_OR_MOVABLE(CudaDevice);

  public:
    CudaDevice(CudaContextHandle, MemoryId affinity_id);
    ~CudaDevice() noexcept final;

    CudaContextHandle context_handle() const {
        return m_context;
    }

    CUcontext context() const {
        return m_context;
    }

    CUstream stream() const {
        return m_stream;
    }

    CUevent event() const {
        return m_event;
    }

    void synchronize() const;

    template<typename... Args>
    void launch(
        dim3 grid_dim,
        dim3 block_dim,
        unsigned int shared_mem,
        void (*const kernel_function)(Args...),
        Args... args) const {
        // Get void pointer to the arguments.
        void* void_args[sizeof...(Args) + 1] = {static_cast<void*>(&args)..., nullptr};

        // Launch the kernel!
        KMM_CUDA_CHECK(cudaLaunchKernel(
            reinterpret_cast<const void*>(kernel_function),
            grid_dim,
            block_dim,
            void_args,
            shared_mem,
            m_stream));
    }

  private:
    CudaContextHandle m_context;
    CUstream m_stream;
    CUevent m_event;
};

/**
 * Contains a handle to the CUDA device thread and allows tasks to be submitted onto the thread.
 */
class CudaDeviceHandle: public DeviceHandle {
  public:
    CudaDeviceHandle(CudaContextHandle context, MemoryId affinity_id, size_t num_streams = 4);
    ~CudaDeviceHandle() noexcept;

    std::unique_ptr<DeviceInfo> info() const override;
    void submit(std::shared_ptr<Task> task, TaskContext context, Completion completion)
        const override;

    class Job: public WorkQueue<Job>::JobBase {
      public:
        Job(std::shared_ptr<Task> task, TaskContext context, Completion completion);

        std::shared_ptr<Task> task;
        TaskContext context;
        Completion completion;
    };

  private:
    CudaDeviceInfo m_info;
    std::shared_ptr<WorkQueue<Job>> m_queue;
    std::thread m_thread;
};

}  // namespace kmm

#endif  // KMM_USE_CUDA
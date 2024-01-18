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

class CudaDeviceInfo: public DeviceInfo {
  public:
    static constexpr size_t NUM_ATTRIBUTES = 19;
    static CUdevice_attribute ATTRIBUTES[NUM_ATTRIBUTES];

    CudaDeviceInfo(CudaContextHandle context, MemoryId affinity_id);

    std::string name() const override {
        return m_name;
    }

    MemoryId memory_affinity() const override {
        return m_affinity_id;
    }

    CUdevice device() const {
        return m_device_id;
    }

    int attribute(CUdevice_attribute attrib) const;

  private:
    std::string m_name;
    CUdevice m_device_id;
    MemoryId m_affinity_id;
    std::array<int, NUM_ATTRIBUTES> m_attributes;
};

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
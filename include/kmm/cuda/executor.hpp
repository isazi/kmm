#pragma once

#include <condition_variable>
#include <cuda.h>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "kmm/cuda/types.hpp"
#include "kmm/executor.hpp"
#include "kmm/host/work_queue.hpp"
#include "kmm/identifiers.hpp"

namespace kmm {

class CudaExecutorInfo: public ExecutorInfo {
  public:
    static constexpr size_t NUM_ATTRIBUTES = 19;
    static CUdevice_attribute ATTRIBUTES[NUM_ATTRIBUTES];

    CudaExecutorInfo(CudaContextHandle context, MemoryId affinity_id);

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

class CudaExecutorContext final: public ExecutorContext, public CudaExecutorInfo {
  public:
    CudaExecutorContext(CudaContextHandle, MemoryId affinity_id);
    ~CudaExecutorContext() noexcept final;

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

  private:
    CudaContextHandle m_context;
    CUstream m_stream;
    CUevent m_event;
};

class CudaExecutor: public Executor {
  public:
    class Job: public WorkQueue<Job>::JobBase {
      public:
        Job(std::shared_ptr<Task> task, TaskContext context, Completion completion);

        std::shared_ptr<Task> task;
        TaskContext context;
        Completion completion;
    };

    CudaExecutor(CudaContextHandle context, MemoryId affinity_id, size_t num_streams = 1);
    ~CudaExecutor() noexcept;

    std::unique_ptr<ExecutorInfo> info() const override;
    void submit(std::shared_ptr<Task> task, TaskContext context, Completion completion)
        const override;

  private:
    CudaExecutorInfo m_info;
    std::shared_ptr<WorkQueue<Job>> m_queue;
    std::thread m_thread;
};

}  // namespace kmm
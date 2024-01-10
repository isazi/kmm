#pragma once

#include <memory>
#include <thread>

#include "kmm/executor.hpp"
#include "kmm/host/thread_pool.hpp"
#include "kmm/task_serialize.hpp"

namespace kmm {

class HostExecutorInfo final: public ExecutorInfo {
    std::string name() const override;
    MemoryId memory_affinity() const override;
};

class ParallelExecutorHandle: public ExecutorHandle, public ThreadPool {
  public:
    std::unique_ptr<ExecutorInfo> info() const override;
    void submit(std::shared_ptr<Task>, TaskContext, Completion) const override;
};

}  // namespace kmm
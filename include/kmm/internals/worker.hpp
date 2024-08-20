#pragma once

#include "kmm/internals/cuda_stream_manager.hpp"
#include "kmm/internals/memory_manager.hpp"
#include "kmm/internals/task_manager.hpp"

namespace kmm {

class Scheduler {
  private:
    std::shared_ptr<MemoryManager> m_memory;
    std::shared_ptr<CudaStreamManager> m_streams;
    std::shared_ptr<TaskManager> m_tasks;
};

}  // namespace kmm
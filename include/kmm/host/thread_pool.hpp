#pragma once

#include <memory>
#include <thread>

#include "kmm/executor.hpp"
#include "kmm/host/work_queue.hpp"
#include "kmm/utils/completion.hpp"

namespace kmm {

class ParallelExecutor;

class ThreadPool {
    class FillJob;
    class CopyJob;
    class ExecutionJob;

  public:
    class Job: public WorkQueue<Job>::JobBase {
      public:
        virtual ~Job() = default;
        virtual void execute(ParallelExecutor&) = 0;
    };

    ThreadPool();
    ~ThreadPool();

    void submit_task(  //
        std::shared_ptr<Task> task,
        TaskContext context,
        Completion completion) const;

    void submit_fill(
        void* dst_data,
        size_t num_bytes,
        std::vector<uint8_t> fill_pattern,
        Completion completion);

    void submit_copy(  //
        const void* src_data,
        void* dst_data,
        size_t num_bytes,
        Completion completion);

  private:
    std::shared_ptr<WorkQueue<Job>> m_queue;
    std::thread m_thread;
};

}  // namespace kmm
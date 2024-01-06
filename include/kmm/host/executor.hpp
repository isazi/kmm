#pragma once

#include <memory>
#include <thread>

#include "kmm/executor.hpp"
#include "kmm/host/thread_pool.hpp"
#include "kmm/task_serialize.hpp"

namespace kmm {

class HostExecutorInfo: public ExecutorInfo {
    std::string name() const override;
    MemoryId memory_affinity() const override;
};

class ParallelExecutor: public Executor, public ThreadPool {
  public:
    std::unique_ptr<ExecutorInfo> info() const override;
    void submit(std::shared_ptr<Task>, TaskContext, Completion) const override;
};

struct Host {
    template<typename Fun, typename... Args>
    EventId operator()(RuntimeImpl& rt, Fun&& fun, Args&&... args) const {
        for (size_t i = 0, n = rt.num_executors(); i < n; i++) {
            auto id = ExecutorId(i);

            if (dynamic_cast<const HostExecutorInfo*>(&rt.executor_info(id)) != nullptr) {
                return TaskLauncher<ExecutionSpace::Host, Fun, Args...>::call(id, rt, fun, args...);
            }
        }

        throw std::runtime_error("could not find host executor");
    }
};

}  // namespace kmm
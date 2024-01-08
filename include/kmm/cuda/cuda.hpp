#pragma once

#include "kmm/cuda/executor.hpp"
#include "kmm/task_serialize.hpp"

namespace kmm {

struct Cuda {
    Cuda(int device = 0) : m_device(device) {}

    template<typename Fun, typename... Args>
    EventId operator()(RuntimeImpl& rt, Fun&& fun, Args&&... args) const {
        for (size_t i = 0, n = rt.num_executors(); i < n; i++) {
            auto id = ExecutorId(i);
            auto* info = dynamic_cast<const CudaExecutorInfo*>(&rt.executor_info(id));

            if (info != nullptr && info->device() == m_device) {
                return TaskLauncher<ExecutionSpace::Cuda, Fun, Args...>::call(id, rt, fun, args...);
            }
        }

        throw std::runtime_error("could not find cuda executor");
    }

  private:
    int m_device;
};

}  // namespace kmm

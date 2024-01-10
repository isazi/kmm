#pragma once
#include "executor.hpp"

#include "kmm/executor.hpp"

namespace kmm {

struct HostLauncher {
    static constexpr ExecutionSpace execution_space = ExecutionSpace::Host;

    template<typename F, typename... Args>
    void operator()(Executor&, TaskContext&, F&& fun, Args&&... args) const {
        std::forward<F>(fun)(std::forward<Args>(args)...);
    }
};

struct Host {
    template<typename Fun, typename... Args>
    EventId operator()(RuntimeImpl& rt, Fun&& fun, Args&&... args) const {
        for (size_t i = 0, n = rt.num_executors(); i < n; i++) {
            auto id = ExecutorId(i);

            if (dynamic_cast<const HostExecutorInfo*>(&rt.executor_info(id)) != nullptr) {
                return TaskLauncher<HostLauncher, Fun, Args...>::call(
                    HostLauncher {},
                    id,
                    rt,
                    fun,
                    args...);
            }
        }

        throw std::runtime_error("could not find host executor");
    }
};

}  // namespace kmm
#pragma once
#include "executor.hpp"

#include "kmm/executor.hpp"

namespace kmm {

struct Host {
    static constexpr ExecutionSpace execution_space = ExecutionSpace::Host;

    ExecutorId find_executor(RuntimeImpl& rt) const {
        for (size_t i = 0, n = rt.num_executors(); i < n; i++) {
            auto id = ExecutorId(i);

            if (dynamic_cast<const HostExecutorInfo*>(&rt.executor_info(id)) != nullptr) {
                return id;
            }
        }

        throw std::runtime_error("could not find host executor");
    }

    template<typename F, typename... Args>
    void operator()(Executor&, TaskContext&, F&& fun, Args&&... args) const {
        std::forward<F>(fun)(std::forward<Args>(args)...);
    }
};

}  // namespace kmm
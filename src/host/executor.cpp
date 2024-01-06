#include <algorithm>
#include <utility>

#include "kmm/host/executor.hpp"
#include "kmm/panic.hpp"

namespace kmm {

std::string HostExecutorInfo::name() const {
    return "CPU";
}

MemoryId HostExecutorInfo::memory_affinity() const {
    return MemoryId(0);
}

std::unique_ptr<ExecutorInfo> ParallelExecutor::info() const {
    return std::make_unique<HostExecutorInfo>();
}

void ParallelExecutor::submit(
    std::shared_ptr<Task> task,
    TaskContext context,
    Completion completion) const {
    this->submit_task(  //
        std::move(task),
        std::move(context),
        std::move(completion));
}

}  // namespace kmm
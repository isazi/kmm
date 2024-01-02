#include "spdlog/spdlog.h"

#include "kmm/executor.hpp"

namespace kmm {

TaskCompletion::TaskCompletion(std::shared_ptr<ITaskCompletion> impl) : m_impl(std::move(impl)) {}

void TaskCompletion::complete(Result<void> result) {
    spdlog::debug("TaskCompletion::complete {}", size_t(m_impl.get()));
    if (auto inner = std::exchange(m_impl, {})) {
        inner->complete_task(std::move(result));
    }
}

TaskCompletion::~TaskCompletion() {
    if (m_impl) {
        complete(ErrorPtr("tasks was not completed properly"));
    }
}

}  // namespace kmm
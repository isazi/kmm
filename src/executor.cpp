#include "kmm/executor.hpp"

namespace kmm {

TaskCompletion::TaskCompletion(std::shared_ptr<Impl> impl) : m_impl(std::move(impl)) {}

void TaskCompletion::complete(TaskResult result) {
    if (auto inner = std::exchange(m_impl, {})) {
        inner->complete_task(std::move(result));
    }
}

void TaskCompletion::complete_err(const std::string& error) {
    complete(TaskError(error));
}

TaskCompletion::~TaskCompletion() {
    if (m_impl) {
        complete_err("tasks was not completed properly");
    }
}

}  // namespace kmm
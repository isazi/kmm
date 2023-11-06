#include "spdlog/spdlog.h"

#include "kmm/scheduler_thread.hpp"
#include "kmm/utils.hpp"

namespace kmm {
void run_forever(std::shared_ptr<Scheduler> scheduler) {
    while (!scheduler->has_shutdown()) {
        auto deadline = std::chrono::system_clock::now() + std::chrono::seconds();
        scheduler->make_progress(deadline);
    }
}

SchedulerThread::SchedulerThread(std::shared_ptr<Scheduler> scheduler) {
    m_thread = std::thread(run_forever, std::move(scheduler));
}

void SchedulerThread::join() {
    if (m_thread.joinable()) {
        m_thread.join();
    }
}

SchedulerThread::~SchedulerThread() {
    join();
}
}  // namespace kmm
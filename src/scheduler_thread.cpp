#include "kmm/scheduler_thread.hpp"

#include "kmm/utils.hpp"

namespace kmm {
SchedulerThread::SchedulerThread(std::shared_ptr<Scheduler> scheduler) :
    m_scheduler(std::move(scheduler)) {}

void SchedulerThread::launch() {
    KMM_ASSERT(m_thread.joinable() == false);

    m_thread = std::thread([scheduler = m_scheduler] {
        while (true) {
            auto deadline = std::chrono::system_clock::now() + std::chrono::seconds();
            scheduler->make_progress(deadline);
        }
    });
}

SchedulerThread::~SchedulerThread() {
    m_thread.join();
}
}  // namespace kmm
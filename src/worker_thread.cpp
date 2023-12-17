#include "spdlog/spdlog.h"

#include "kmm/utils.hpp"
#include "kmm/worker_thread.hpp"

namespace kmm {
void run_forever(std::shared_ptr<Worker> scheduler) {
    while (!scheduler->has_shutdown()) {
        auto deadline = std::chrono::system_clock::now() + std::chrono::seconds();
        scheduler->make_progress(deadline);
    }
}

WorkerThread::WorkerThread(std::shared_ptr<Worker> scheduler) {
    m_thread = std::thread(run_forever, std::move(scheduler));
}

void WorkerThread::join() {
    if (m_thread.joinable()) {
        m_thread.join();
    }
}

WorkerThread::~WorkerThread() {
    join();
}
}  // namespace kmm
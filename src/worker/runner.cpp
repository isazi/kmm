#include "spdlog/spdlog.h"

#include "kmm/worker/runner.hpp"
#include "kmm/worker/worker.hpp"

namespace kmm {
void run_forever(std::shared_ptr<Worker> worker) {
    while (!worker->is_shutdown()) {
        auto deadline = std::chrono::system_clock::now() + std::chrono::milliseconds(100);
        worker->make_progress(deadline);
    }
}

WorkerRunner::WorkerRunner(std::shared_ptr<Worker> worker) {
    m_thread = std::thread(run_forever, std::move(worker));
}

void WorkerRunner::join() {
    if (m_thread.joinable()) {
        m_thread.join();
    }
}

WorkerRunner::~WorkerRunner() {
    join();
}
}  // namespace kmm
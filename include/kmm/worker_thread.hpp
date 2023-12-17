#pragma once

#include <memory>
#include <thread>

#include "kmm/worker.hpp"

namespace kmm {

class WorkerThread {
  public:
    WorkerThread(std::shared_ptr<Worker> scheduler);
    ~WorkerThread();
    void join();

  private:
    std::thread m_thread;
};

}  // namespace kmm
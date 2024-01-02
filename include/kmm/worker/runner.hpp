#pragma once

#include <memory>
#include <thread>

namespace kmm {

class Worker;

class WorkerRunner {
  public:
    WorkerRunner(std::shared_ptr<Worker> worker);
    ~WorkerRunner();
    void join();

  private:
    std::thread m_thread;
};

}  // namespace kmm
#pragma once

#include <memory>
#include <thread>

#include "kmm/scheduler.hpp"

namespace kmm {

class SchedulerThread {
  public:
    SchedulerThread(std::shared_ptr<Scheduler> scheduler);
    ~SchedulerThread();
    void join();

  private:
    std::thread m_thread;
};

}  // namespace kmm
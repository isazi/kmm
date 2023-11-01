#pragma once

#include <memory>
#include <thread>

#include "kmm/scheduler.hpp"

namespace kmm {

class SchedulerThread {
    explicit SchedulerThread(std::shared_ptr<Scheduler> scheduler);
    void launch();
    ~SchedulerThread();

  private:
    std::thread m_thread;
    std::shared_ptr<Scheduler> m_scheduler;
};

}  // namespace kmm
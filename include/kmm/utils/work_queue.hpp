#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>

namespace kmm {
template<typename Job>
class WorkQueue {
  public:
    class JobBase {
      public:
        virtual ~JobBase() = default;

      private:
        friend class WorkQueue;
        std::unique_ptr<Job> next;
    };

    void shutdown() {
        std::unique_lock guard {lock};
        has_shutdown = true;
        cond.notify_all();
    }

    void push(std::unique_ptr<Job> unit) {
        static_assert(std::is_base_of<JobBase, Job>(), "`Job` must extend `JobBase`");
        std::unique_lock guard {lock};

        if (front) {
            as_base(*unit).next = nullptr;
            as_base(*back).next = std::move(unit);
            back = as_base(*back).next.get();
        } else {
            front = std::move(unit);
            back = front.get();
            cond.notify_all();
        }
    }

    std::optional<std::unique_ptr<Job>> pop_wait_until(
        std::chrono::system_clock::time_point deadline) {
        std::unique_lock guard {lock};

        while (true) {
            if (triggered) {
                triggered = false;
                break;
            }

            if (has_shutdown) {
                break;
            }

            if (cond.wait_until(guard, deadline) == std::cv_status::timeout) {
                break;
            }
        }

        if (front == nullptr) {
            return std::nullopt;
        }

        auto popped = std::move(front);
        if (as_base(*popped).next != nullptr) {
            front = std::move(as_base(*popped).next);
        } else {
            front = nullptr;
            back = nullptr;
        }

        return popped;
    }

    std::optional<std::unique_ptr<Job>> pop() {
        return pop_wait_until(std::chrono::system_clock::time_point::max());
    }

    void wakeup() {
        std::unique_lock guard {lock};
        triggered = true;
        cond.notify_all();
    }

  private:
    JobBase& as_base(JobBase& job) {
        return job;
    }

    std::mutex lock;
    std::condition_variable cond;
    bool triggered = false;
    bool has_shutdown = false;
    std::unique_ptr<Job> front = nullptr;
    Job* back = nullptr;
};
}  // namespace kmm
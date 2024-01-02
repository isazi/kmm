#pragma once

#include <memory>
#include <mutex>

namespace kmm {
template<typename Context>
class WorkQueue {
  public:
    class Job {
      public:
        virtual ~Job() = default;
        virtual void execute(Context&) = 0;

      private:
        friend WorkQueue<Context>;
        std::unique_ptr<Job> next;
    };

    void shutdown() {
        std::lock_guard<std::mutex> guard(lock);
        has_shutdown = true;
        cond.notify_all();
    }

    void push(std::unique_ptr<Job> unit) {
        std::lock_guard<std::mutex> guard(lock);
        if (front) {
            unit->next = nullptr;
            back->next = std::move(unit);
            back = back->next.get();
        } else {
            front = std::move(unit);
            back = front.get();
            cond.notify_all();
        }
    }

    void process_forever(Context& context) {
        std::unique_lock<std::mutex> guard(lock);

        while (!has_shutdown || front != nullptr) {
            if (front == nullptr) {
                cond.wait(guard);
                continue;
            }

            auto popped = std::move(front);
            if (popped->next != nullptr) {
                front = std::move(popped->next);
            } else {
                front = nullptr;
                back = nullptr;
            }

            guard.unlock();
            popped->execute(context);
            guard.lock();
        }
    }

  private:
    std::mutex lock;
    std::condition_variable cond;
    bool has_shutdown = false;
    std::unique_ptr<Job> front = nullptr;
    Job* back = nullptr;
};
}  // namespace kmm
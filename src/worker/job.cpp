#include "kmm/worker/job.hpp"
#include "kmm/worker/worker.hpp"

namespace kmm {

void Job::trigger_wakeup(bool allow_progress) const {
    if (auto owner = worker.lock()) {
        auto self = std::shared_ptr<Job>(shared_from_this(), const_cast<Job*>(this));
        owner->wakeup(self, allow_progress);
    }
}

bool JobQueue::is_empty() const {
    return m_head == nullptr;
}

bool JobQueue::push(std::shared_ptr<Job> op) {
    if (op->in_queue.test_and_set()) {
        return false;
    }

    op->next_queue_item = nullptr;

    if (m_head == nullptr) {
        m_tail = op.get();
        m_head = std::move(op);
    } else {
        auto* old_tail = std::exchange(m_tail, op.get());
        old_tail->next_queue_item = std::move(op);
    }

    return true;
}

void JobQueue::push_all(JobQueue that) {
    // the other queue is empty
    if (that.m_head == nullptr) {
        return;
    }

    // This queue is empty, simply copy the head and tail
    if (m_head == nullptr) {
        m_head = std::exchange(that.m_head, nullptr);
        m_tail = std::exchange(that.m_tail, nullptr);
        return;
    }

    // Set next of `this.tail` to head of `that` and simply copy the tail from `that`
    m_tail->next_queue_item = std::exchange(that.m_head, nullptr);
    m_tail = std::exchange(that.m_tail, nullptr);
}

std::optional<std::shared_ptr<Job>> JobQueue::pop() {
    // This queue is empty
    if (m_head == nullptr) {
        return std::nullopt;
    }

    auto op = std::move(m_head);
    m_head = std::move(op->next_queue_item);

    op->in_queue.clear();
    return std::optional {std::move(op)};
}

bool SharedJobQueue::push(std::shared_ptr<Job> op) const {
    std::lock_guard guard {m_lock};
    auto id = op->id();
    bool needs_notify = m_queue.is_empty();

    if (m_queue.push(std::move(op))) {
        spdlog::debug("pushed id={}, needs_notify={}", id, needs_notify);
        if (needs_notify) {
            m_cond.notify_one();
        }

        return true;
    }

    return false;
}

JobQueue SharedJobQueue::pop_all(
    std::chrono::time_point<std::chrono::system_clock> deadline) const {
    std::unique_lock guard {m_lock};
    if (deadline != std::chrono::time_point<std::chrono::system_clock> {}) {
        while (m_queue.is_empty()) {
            if (m_cond.wait_until(guard, deadline) == std::cv_status::timeout) {
                break;
            }
        }
    }

    return std::move(m_queue);
}

}  // namespace kmm
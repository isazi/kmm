#include <cstring>
#include <utility>

#include "kmm/platforms/host.hpp"
#include "kmm/scheduler.hpp"
#include "kmm/utils.hpp"

namespace kmm {

struct ParallelExecutor::Job {
    virtual ~Job() = default;
    virtual void execute(ParallelExecutorContext&) = 0;
    std::unique_ptr<Job> next;
};

struct ParallelExecutor::Queue {
    std::mutex lock;
    std::condition_variable cond;
    bool has_shutdown = false;
    std::unique_ptr<Job> front = nullptr;
    Job* back = nullptr;

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

    void process_forever() {
        std::unique_lock<std::mutex> guard(lock);
        ParallelExecutorContext context {};

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
};

ParallelExecutor::ParallelExecutor() :
    m_queue(std::make_shared<Queue>()),
    m_thread([q = m_queue] { q->process_forever(); }) {
    // Do we want to detach or join?
    m_thread.detach();
}

ParallelExecutor::~ParallelExecutor() = default;

class ExecuteJob: public ParallelExecutor::Job {
  public:
    ExecuteJob(std::shared_ptr<Task>&& task, TaskContext&& context, TaskCompletion&& completion) :
        m_task(std::move(task)),
        m_context(std::move(context)),
        m_completion(std::move(completion)) {}

    void execute(ParallelExecutorContext& executor) override {
        try {
            m_completion.complete(m_task->execute(executor, m_context));
        } catch (const std::exception& e) {
            m_completion.complete_err(e.what());
        }
    }

  private:
    std::shared_ptr<Task> m_task;
    TaskContext m_context;
    TaskCompletion m_completion;
};

void ParallelExecutor::submit(
    std::shared_ptr<Task> task,
    TaskContext context,
    TaskCompletion completion) {
    m_queue->push(
        std::make_unique<ExecuteJob>(std::move(task), std::move(context), std::move(completion)));
}

class CopyJob: public ParallelExecutor::Job {
  public:
    CopyJob(
        const void* src_ptr,
        void* dst_ptr,
        size_t nbytes,
        std::unique_ptr<MemoryCompletion>&& completion) :
        src_ptr(src_ptr),
        dst_ptr(dst_ptr),
        nbytes(nbytes),
        completion(std::move(completion)) {}

    void execute(ParallelExecutorContext&) override {
        std::memcpy(dst_ptr, src_ptr, nbytes);
        completion->complete();
    }

  private:
    const void* src_ptr;
    void* dst_ptr;
    size_t nbytes;
    std::unique_ptr<MemoryCompletion> completion;
};

void ParallelExecutor::copy_async(
    const void* src_ptr,
    void* dst_ptr,
    size_t nbytes,
    std::unique_ptr<MemoryCompletion> completion) const {
    m_queue->push(std::make_unique<CopyJob>(src_ptr, dst_ptr, nbytes, std::move(completion)));
}

HostAllocation::HostAllocation(size_t nbytes) : m_nbytes(nbytes) {
    m_data = std::make_unique<char[]>(nbytes);
}

HostMemory::HostMemory(std::shared_ptr<ParallelExecutor> executor, size_t max_bytes) :
    m_executor(std::move(executor)),
    m_bytes_remaining(max_bytes) {}

std::optional<std::unique_ptr<MemoryAllocation>> HostMemory::allocate(
    DeviceId id,
    size_t num_bytes) {
    KMM_ASSERT(id == 0);
    if (m_bytes_remaining >= num_bytes) {
        m_bytes_remaining -= num_bytes;
        return std::make_unique<HostAllocation>(num_bytes);
    } else {
        return std::nullopt;
    }
}

void HostMemory::deallocate(DeviceId id, std::unique_ptr<MemoryAllocation> allocation) {
    KMM_ASSERT(id == 0);
    auto& alloc = dynamic_cast<HostAllocation&>(*allocation);
    m_bytes_remaining += alloc.size();
}

bool HostMemory::is_copy_possible(DeviceId src_id, DeviceId dst_id) {
    return src_id == 0 && dst_id == 0;
}

void HostMemory::copy_async(
    DeviceId src_id,
    const MemoryAllocation* src_alloc,
    size_t src_offset,
    DeviceId dst_id,
    const MemoryAllocation* dst_alloc,
    size_t dst_offset,
    size_t num_bytes,
    std::unique_ptr<MemoryCompletion> completion) {
    KMM_ASSERT(src_id == 0);
    KMM_ASSERT(dst_id == 0);

    const auto& src_host = dynamic_cast<const HostAllocation&>(*src_alloc);
    const auto& dst_host = dynamic_cast<const HostAllocation&>(*dst_alloc);

    KMM_ASSERT(num_bytes <= src_host.size());
    KMM_ASSERT(num_bytes <= dst_host.size());

    KMM_ASSERT(src_offset <= src_host.size() - num_bytes);
    KMM_ASSERT(dst_offset <= dst_host.size() - num_bytes);

    m_executor->copy_async(
        static_cast<const char*>(src_host.data()) + src_offset,
        static_cast<char*>(dst_host.data()) + dst_offset,
        num_bytes,
        std::move(completion));
}

}  // namespace kmm
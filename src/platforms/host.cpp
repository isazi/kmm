#include <cstring>
#include <utility>

#include "kmm/platforms/host.hpp"
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

class ExecutionJob: public ParallelExecutor::Job {
  public:
    ExecutionJob(std::shared_ptr<Task>&& task, TaskContext&& context, TaskCompletion&& completion) :
        m_task(std::move(task)),
        m_context(std::move(context)),
        m_completion(std::move(completion)) {}

    void execute(ParallelExecutorContext& executor) override {
        try {
            m_task->execute(executor, m_context);
            m_completion.complete();
        } catch (...) {
            m_completion.complete(ErrorPtr::from_current_exception());
        }
    }

  private:
    std::shared_ptr<Task> m_task;
    TaskContext m_context;
    TaskCompletion m_completion;
};

void ParallelExecutor::submit_job(std::unique_ptr<Job> job) {
    m_queue->push(std::move(job));
}

void ParallelExecutor::submit(
    std::shared_ptr<Task> task,
    TaskContext context,
    TaskCompletion completion) {
    m_queue->push(
        std::make_unique<ExecutionJob>(std::move(task), std::move(context), std::move(completion)));
}

class CopyJob: public ParallelExecutor::Job {
  public:
    CopyJob(const void* src_ptr, void* dst_ptr, size_t nbytes, MemoryCompletion&& completion) :
        src_ptr(src_ptr),
        dst_ptr(dst_ptr),
        nbytes(nbytes),
        completion(std::move(completion)) {}

    void execute(ParallelExecutorContext&) override {
        std::memcpy(dst_ptr, src_ptr, nbytes);
        completion.complete(Result<void>());
    }

  private:
    const void* src_ptr;
    void* dst_ptr;
    size_t nbytes;
    MemoryCompletion completion;
};

class FillJob: public ParallelExecutor::Job {
  public:
    FillJob(
        void* dst_ptr,
        size_t nbytes,
        std::vector<uint8_t>&& fill_bytes,
        MemoryCompletion&& completion) :
        dst_ptr(dst_ptr),
        nbytes(nbytes),
        fill_bytes(std::move(fill_bytes)),
        completion(std::move(completion)) {}

    void execute(ParallelExecutorContext&) override {
        size_t k = fill_bytes.size();

        if (k == 1) {
            fill_impl(std::integral_constant<size_t, 1>());
        } else if (k == 2) {
            fill_impl(std::integral_constant<size_t, 2>());
        } else if (k == 4) {
            fill_impl(std::integral_constant<size_t, 4>());
        } else if (k == 8) {
            fill_impl(std::integral_constant<size_t, 8>());
        } else if (k == 12) {
            fill_impl(std::integral_constant<size_t, 12>());
        } else if (k == 16) {
            fill_impl(std::integral_constant<size_t, 16>());
        } else {
            fill_impl(k);
        }

        completion.complete(Result<void>());
    }

    template<typename K>
    void fill_impl(K k) {
        size_t i = 0;

        for (; i < nbytes / k; i++) {
            for (size_t j = 0; j < k; j++) {
                static_cast<uint8_t*>(dst_ptr)[i * k + j] = fill_bytes[j];
            }
        }

        for (size_t j = 0; j < nbytes % k; j++) {
            static_cast<uint8_t*>(dst_ptr)[i * k + j] = fill_bytes[j];
        }
    }

  private:
    void* dst_ptr;
    size_t nbytes;
    std::vector<uint8_t> fill_bytes;
    MemoryCompletion completion;
};

HostAllocation::HostAllocation(size_t nbytes) : m_nbytes(nbytes) {
    m_data = std::unique_ptr<char[]>(new char[nbytes]);
}

HostMemory::HostMemory(std::shared_ptr<ParallelExecutor> executor, size_t max_bytes) :
    m_executor(std::move(executor)),
    m_bytes_remaining(max_bytes) {}

std::optional<std::unique_ptr<MemoryAllocation>> HostMemory::allocate(
    MemoryId id,
    size_t num_bytes) {
    KMM_ASSERT(id == 0);
    if (m_bytes_remaining >= num_bytes) {
        m_bytes_remaining -= num_bytes;
        return std::make_unique<HostAllocation>(num_bytes);
    } else {
        return std::nullopt;
    }
}

void HostMemory::deallocate(MemoryId id, std::unique_ptr<MemoryAllocation> allocation) {
    KMM_ASSERT(id == 0);
    auto& alloc = dynamic_cast<HostAllocation&>(*allocation);
    m_bytes_remaining += alloc.size();
}

bool HostMemory::is_copy_possible(MemoryId src_id, MemoryId dst_id) {
    return src_id == 0 && dst_id == 0;
}

void HostMemory::copy_async(
    MemoryId src_id,
    const MemoryAllocation* src_alloc,
    size_t src_offset,
    MemoryId dst_id,
    const MemoryAllocation* dst_alloc,
    size_t dst_offset,
    size_t num_bytes,
    MemoryCompletion completion) {
    const auto& src_host = dynamic_cast<const HostAllocation&>(*src_alloc);
    const auto& dst_host = dynamic_cast<const HostAllocation&>(*dst_alloc);

    KMM_ASSERT(src_id == 0);
    KMM_ASSERT(dst_id == 0);
    KMM_ASSERT(num_bytes <= src_host.size());
    KMM_ASSERT(num_bytes <= dst_host.size());
    KMM_ASSERT(src_offset <= src_host.size() - num_bytes);
    KMM_ASSERT(dst_offset <= dst_host.size() - num_bytes);

    m_executor->submit_job(std::make_unique<CopyJob>(
        static_cast<const char*>(src_host.data()) + src_offset,
        static_cast<char*>(dst_host.data()) + dst_offset,
        num_bytes,
        std::move(completion)));
}

void HostMemory::fill_async(
    MemoryId dst_id,
    const MemoryAllocation* dst_alloc,
    size_t dst_offset,
    size_t num_bytes,
    std::vector<uint8_t> fill_bytes,
    MemoryCompletion completion) {
    const auto& dst_host = dynamic_cast<const HostAllocation&>(*dst_alloc);

    KMM_ASSERT(dst_id == 0);
    KMM_ASSERT(num_bytes <= dst_host.size());
    KMM_ASSERT(dst_offset <= dst_host.size() - num_bytes);

    m_executor->submit_job(std::make_unique<FillJob>(
        static_cast<char*>(dst_host.data()) + dst_offset,
        num_bytes,
        std::move(fill_bytes),
        std::move(completion)));
}

}  // namespace kmm
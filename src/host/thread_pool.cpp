#include "kmm/host/executor.hpp"
#include "kmm/host/thread_pool.hpp"

namespace kmm {

ThreadPool::ThreadPool() :
    m_queue(std::make_shared<WorkQueue<Job>>()),
    m_thread([q = m_queue] {
        ParallelExecutor context {};

        while (auto popped = q->pop()) {
            (*popped)->execute(context);
        }
    }) {
    // Do we want to detach or join?
    m_thread.detach();
}

ThreadPool::~ThreadPool() = default;

class ThreadPool::ExecutionJob: public ThreadPool::Job {
  public:
    ExecutionJob(std::shared_ptr<Task>&& task, TaskContext&& context, Completion&& completion) :
        m_task(std::move(task)),
        m_context(std::move(context)),
        m_completion(std::move(completion)) {}

    void execute(ParallelExecutor& executor) override {
        try {
            m_task->execute(executor, m_context);
            m_completion.complete_ok();
        } catch (...) {
            m_completion.complete(ErrorPtr::from_current_exception());
        }
    }

  private:
    std::shared_ptr<Task> m_task;
    TaskContext m_context;
    Completion m_completion;
};

void ThreadPool::submit_task(std::shared_ptr<Task> task, TaskContext context, Completion completion)
    const {
    m_queue->push(std::make_unique<ExecutionJob>(  //
        std::move(task),
        std::move(context),
        std::move(completion)));
}

class ThreadPool::FillJob: public ThreadPool::Job {
  public:
    FillJob(
        void* dst_ptr,
        size_t num_bytes,
        std::vector<uint8_t>&& fill_bytes,
        Completion&& completion) :
        dst_ptr(dst_ptr),
        num_bytes(num_bytes),
        fill_bytes(std::move(fill_bytes)),
        completion(std::move(completion)) {}

    template<typename K>
    void fill_impl(K k) {
        size_t i = 0;

        for (; i < num_bytes / k; i++) {
            for (size_t j = 0; j < k; j++) {
                static_cast<uint8_t*>(dst_ptr)[i * k + j] = fill_bytes[j];
            }
        }

        for (size_t j = 0; j < num_bytes % k; j++) {
            static_cast<uint8_t*>(dst_ptr)[i * k + j] = fill_bytes[j];
        }
    }

    void execute(ParallelExecutor&) override {
        size_t k = fill_bytes.size();

        if (k == 1) {
            std::memset(dst_ptr, fill_bytes[0], num_bytes);
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

  private:
    void* dst_ptr;
    size_t num_bytes;
    std::vector<uint8_t> fill_bytes;
    Completion completion;
};

void ThreadPool::submit_fill(
    void* dst_data,
    size_t num_bytes,
    std::vector<uint8_t> fill_pattern,
    Completion completion) {
    m_queue->push(std::make_unique<FillJob>(
        dst_data,
        num_bytes,
        std::move(fill_pattern),
        std::move(completion)));
}

class ThreadPool::CopyJob: public ThreadPool::Job {
  public:
    CopyJob(const void* src_ptr, void* dst_ptr, size_t num_bytes, Completion&& completion) :
        src_ptr(src_ptr),
        dst_ptr(dst_ptr),
        num_bytes(num_bytes),
        completion(std::move(completion)) {}

    void execute(ParallelExecutor&) override {
        std::memcpy(dst_ptr, src_ptr, num_bytes);
        completion.complete_ok();
    }

  private:
    const void* src_ptr;
    void* dst_ptr;
    size_t num_bytes;
    Completion completion;
};

void ThreadPool::submit_copy(
    const void* src_data,
    void* dst_data,
    size_t num_bytes,
    Completion completion) {
    m_queue->push(std::make_unique<CopyJob>(src_data, dst_data, num_bytes, std::move(completion)));
}

}  // namespace kmm
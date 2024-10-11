#pragma once

#include <mutex>

#include "buffer_manager.hpp"
#include "cuda_stream_manager.hpp"
#include "executor.hpp"
#include "memory_manager.hpp"
#include "scheduler.hpp"
#include "task_graph.hpp"

#include "kmm/core/system_info.hpp"

namespace kmm {

class BufferGuard;

class Worker: public std::enable_shared_from_this<Worker> {
    KMM_NOT_COPYABLE_OR_MOVABLE(Worker)

  public:
    Worker(std::vector<CudaContextHandle> contexts);
    ~Worker();

    bool query_event(EventId event_id, std::chrono::system_clock::time_point deadline);
    void make_progress();
    void shutdown();

    BufferGuard access_buffer(
        BufferId buffer_id,
        MemoryId memory_id = MemoryId::host(),
        AccessMode mode = AccessMode::Exclusive
    );

    template<typename F>
    auto with_task_graph(F fun) {
        std::lock_guard guard {m_mutex};

        if (m_has_shutdown) {
            throw std::runtime_error("cannot submit work, worker has been shut down");
        }

        if constexpr (std::is_void<decltype(fun(*m_graph))>::value) {
            fun(*m_graph);
            m_graph->commit();
        } else {
            auto result = fun(*m_graph);
            m_graph->commit();
            return result;
        }
    }

    const SystemInfo& system_info() const {
        return m_info;
    }

  private:
    friend class BufferGuardTracker;

    void flush_events_impl();
    void make_progress_impl();
    void execute_command(std::shared_ptr<TaskNode> node);

    mutable std::mutex m_mutex;
    mutable bool m_has_shutdown = false;
    SystemInfo m_info;
    std::shared_ptr<Scheduler> m_scheduler;
    std::shared_ptr<CudaStreamManager> m_streams;
    std::shared_ptr<MemoryManager> m_memory;
    std::shared_ptr<BufferManager> m_buffers;
    std::shared_ptr<Executor> m_executor;
    std::shared_ptr<TaskGraph> m_graph;
    std::shared_ptr<MemoryManager::Transaction> m_root_transaction;
};

std::shared_ptr<Worker> make_worker();

}  // namespace kmm
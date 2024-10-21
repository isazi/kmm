#pragma once

#include <mutex>

#include "executor.hpp"
#include "scheduler.hpp"
#include "task_graph.hpp"

#include "kmm/core/system_info.hpp"

namespace kmm {

class BufferGuard;

class Worker: public std::enable_shared_from_this<Worker> {
    KMM_NOT_COPYABLE_OR_MOVABLE(Worker)

  public:
    Worker(
        std::vector<CudaContextHandle> contexts,
        std::shared_ptr<CudaStreamManager> stream_manager,
        std::shared_ptr<MemoryManager> memory_system
    );
    ~Worker();

    bool query_event(EventId event_id, std::chrono::system_clock::time_point deadline);
    bool is_idle();
    void make_progress();
    void shutdown();

    template<typename F>
    auto with_task_graph(F fun) {
        std::lock_guard guard {m_mutex};

        if (m_has_shutdown) {
            throw std::runtime_error("cannot submit work, worker has been shut down");
        }

        try {
            if constexpr (std::is_void<decltype(fun(m_graph))>::value) {
                fun(m_graph);
                m_graph.commit();
            } else {
                auto result = fun(m_graph);
                m_graph.commit();
                return result;
            }
        } catch (...) {
            m_graph.rollback();
            throw;
        }
    }

    const SystemInfo& system_info() const {
        return m_info;
    }

  private:
    friend class BufferGuardTracker;

    void flush_events_impl();
    void make_progress_impl();
    bool is_idle_impl();

    mutable std::mutex m_mutex;
    mutable bool m_has_shutdown = false;
    SystemInfo m_info;
    Executor m_executor;
    TaskGraph m_graph;

    std::shared_ptr<CudaStreamManager> m_stream_manager;
    std::shared_ptr<MemoryManager> m_memory_manager;
};

std::shared_ptr<Worker> make_worker();

}  // namespace kmm
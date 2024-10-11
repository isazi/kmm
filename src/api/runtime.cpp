#include "kmm/api/runtime.hpp"
#include "kmm/core/cuda_device.hpp"
#include "kmm/internals/worker.hpp"

namespace kmm {

class CopyInTask: public Task {
  public:
    CopyInTask(const void* data, size_t nbytes) : m_src_addr(data), m_nbytes(nbytes) {}

    void execute(ExecutionContext& proc, TaskContext context) override {
        KMM_ASSERT(context.accessors.size() == 1);
        KMM_ASSERT(context.accessors[0].layout.size_in_bytes == m_nbytes);

        void* dst_addr = context.accessors[0].address;

        if (auto* device = proc.cast_if<CudaDevice>()) {
            device->copy_bytes(m_src_addr, dst_addr, m_nbytes);
        } else if (proc.is<HostContext>()) {
            ::memcpy(dst_addr, m_src_addr, m_nbytes);
        } else {
            throw std::runtime_error("invalid execution context");
        }
    }

  private:
    const void* m_src_addr;
    size_t m_nbytes;
};

Runtime::Runtime(std::shared_ptr<Worker> worker) : m_worker(std::move(worker)) {
    KMM_ASSERT(m_worker != nullptr);
}

Runtime::Runtime(Worker& worker) : Runtime(worker.shared_from_this()) {}

MemoryId Runtime::memory_affinity_for_address(const void* address) const {
    if (auto device_opt = get_cuda_device_by_address(address)) {
        const auto& device = m_worker->system_info().device_by_ordinal(*device_opt);
        return device.memory_id();
    } else {
        return MemoryId::host();
    }
}

BufferId Runtime::allocate_bytes(const void* data, BufferLayout layout, MemoryId memory_id) {
    BufferId buffer_id;
    EventId event_id = m_worker->with_task_graph([&](TaskGraph& graph) {
        buffer_id = graph.create_buffer(layout);
        auto req = BufferRequirement {
            .buffer_id = buffer_id,
            .memory_id = memory_id,
            .access_mode = AccessMode::Exclusive  //
        };

        auto task = std::make_shared<CopyInTask>(data, layout.size_in_bytes);
        ProcessorId proc = memory_id.is_device() ? memory_id.as_device() : ProcessorId::host();

        return graph.insert_task(proc, task, {req});
    });

    wait(event_id);
    return buffer_id;
}

bool Runtime::is_done(EventId id) const {
    return m_worker->query_event(id, std::chrono::system_clock::time_point::min());
}

void Runtime::wait(EventId id) const {
    m_worker->query_event(id, std::chrono::system_clock::time_point::max());
}

bool Runtime::wait_until(EventId id, typename std::chrono::system_clock::time_point deadline)
    const {
    return m_worker->query_event(id, deadline);
}

bool Runtime::wait_for(EventId id, typename std::chrono::system_clock::duration duration) const {
    return m_worker->query_event(id, std::chrono::system_clock::now() + duration);
}

EventId Runtime::barrier() const {
    return m_worker->with_task_graph([&](TaskGraph& g) { return g.insert_barrier(); });
}

void Runtime::synchronize() const {
    wait(barrier());
}

const SystemInfo& Runtime::info() const {
    return m_worker->system_info();
}

const Worker& Runtime::worker() const {
    return *m_worker;
}

Runtime make_runtime() {
    return make_worker();
}

}  // namespace kmm
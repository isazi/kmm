#include "kmm/api/runtime.hpp"

namespace kmm {

class CopyInTask: public HostTask, public DeviceTask {
  public:
    CopyInTask(const void* data, size_t nbytes) : m_src_addr(data), m_nbytes(nbytes) {}

    void execute(TaskContext context) override {
        KMM_ASSERT(context.accessors.size() == 1);
        KMM_ASSERT(context.accessors[0].layout.size_in_bytes == m_nbytes);

        void* dst_addr = context.accessors[0].address;
        ::memcpy(dst_addr, m_src_addr, m_nbytes);
    }

    void execute(CudaDevice& device, TaskContext context) override {
        KMM_ASSERT(context.accessors.size() == 1);
        KMM_ASSERT(context.accessors[0].layout.size_in_bytes == m_nbytes);

        void* dst_addr = context.accessors[0].address;
        device.copy_bytes(m_src_addr, dst_addr, m_nbytes);
        device.synchronize();
    }

  private:
    const void* m_src_addr;
    size_t m_nbytes;
};

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

        if (memory_id.is_device()) {
            return graph.insert_device_task(memory_id.as_device(), task, {req});
        } else {
            return graph.insert_host_task(task, {req});
        }
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
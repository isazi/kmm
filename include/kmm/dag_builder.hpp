#pragma once

#include "kmm/executor.hpp"
#include "kmm/scheduler.hpp"

namespace kmm {

class DAGBuilder {
  public:
    BufferId create_buffer(const BufferLayout&);
    void delete_buffer(BufferId);

    OperationId submit_task(
        DeviceId device_id,
        std::shared_ptr<Task> task,
        const std::vector<VirtualBufferRequirement>& buffers,
        std::vector<OperationId> dependencies);

    OperationId submit_barrier();
    OperationId submit_buffer_barrier(BufferId buffer_id);
    OperationId submit_promise(OperationId op_id, std::promise<void> promise);

    std::vector<CommandPacket> flush();
    void flush(Scheduler& scheduler);

  private:
    PhysicalBufferId update_buffer_access(
        BufferId,
        OperationId,
        AccessMode,
        std::vector<OperationId>& deps_out);

    struct Record {
        BufferId virtual_id;
        PhysicalBufferId physical_id;
        std::string name;
        std::vector<OperationId> last_writers;
        std::vector<OperationId> last_readers;
    };

    OperationId m_next_job_id = OperationId(1);
    BufferId m_next_buffer_id = BufferId(1);
    std::unordered_map<BufferId, std::unique_ptr<Record>> m_buffers;
    std::vector<CommandPacket> m_commands;
};

}  // namespace kmm
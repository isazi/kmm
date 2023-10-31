#pragma once

#include "kmm/scheduler.hpp"

namespace kmm {

struct VirtualBufferRequirement {
    BufferId buffer_id;
    AccessMode mode;
};

class DAGBuilder {
  public:
    BufferId create_buffer(const BufferLayout&);
    void delete_buffer(BufferId);

    JobId submit_task(
        DeviceId device_id,
        std::shared_ptr<Task> task,
        const std::vector<VirtualBufferRequirement>& buffers,
        std::vector<JobId> dependencies);

    JobId submit_barrier();

    std::vector<CommandPacket> flush();
    void flush(Scheduler& scheduler);

    JobId submit_buffer_barrier(BufferId identifier);

  private:
    PhysicalBufferId update_buffer_access(
        BufferId,
        JobId,
        AccessMode,
        std::vector<JobId>& deps_out);

    struct Record {
        BufferId virtual_id;
        PhysicalBufferId physical_id;
        std::string name;
        std::vector<JobId> last_writers;
        std::vector<JobId> last_readers;
    };

    JobId m_next_job_id = JobId(1);
    BufferId m_next_buffer_id = BufferId(1);
    std::unordered_map<BufferId, std::unique_ptr<Record>> m_buffers;
    std::vector<CommandPacket> m_commands;
};

}  // namespace kmm
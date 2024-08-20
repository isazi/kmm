#include "kmm/core/buffer.hpp"
#include "kmm/core/copy_specification.hpp"
#include "kmm/core/identifiers.hpp"
#include "kmm/core/task.hpp"

namespace kmm {

class Command {};

struct BufferRequirement {
    BufferId buffer_id;
    MemoryId memory_id;
    AccessMode access;
};

class GraphBuilder {
  public:
    BufferId create_buffer(BufferLayout layout);

    EventId delete_buffer(BufferId buffer);

    EventId prefetch_buffer(BufferId buffer, MemoryId target, EventList deps = {});

    EventId join_events(EventList deps);

    EventId copy_buffer(
        BufferId src_id,
        MemoryId src_memory,
        BufferId dst_id,
        MemoryId dst_memory,
        CopySpecification spec,
        EventList deps = {});

    EventId enqueue_host_task(
        std::shared_ptr<HostTask> task,
        std::vector<BufferRequirement> buffers,
        EventList deps = {});

    EventId enqueue_device_task(
        std::shared_ptr<DeviceTask> task,
        std::vector<BufferRequirement> buffers,
        EventList deps = {});
};

}  // namespace kmm
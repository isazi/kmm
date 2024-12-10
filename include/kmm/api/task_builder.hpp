#pragma once

#include "kmm/core/buffer.hpp"
#include "kmm/core/identifiers.hpp"

namespace kmm {

class TaskGraph;
class Worker;

struct TaskInit {
    std::shared_ptr<Worker>& worker;
    TaskGraph& graph;
    const TaskPartition& partition;
};

struct TaskBuilder {
    std::shared_ptr<Worker>& worker;
    TaskGraph& graph;
    TaskChunk chunk;
    MemoryId memory_id;
    std::vector<BufferRequirement> buffers;
    EventList dependencies;

    size_t add_buffer_requirement(BufferRequirement req) {
        size_t index = buffers.size();
        buffers.push_back(std::move(req));
        return index;
    }
};

struct TaskResult {
    std::shared_ptr<Worker>& worker;
    TaskGraph& graph;
    EventList events;
};

}  // namespace kmm
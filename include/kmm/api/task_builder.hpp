#pragma once

#include "kmm/core/buffer.hpp"
#include "kmm/core/identifiers.hpp"

namespace kmm {

class TaskGraph;
class Worker;

struct TaskBuilder {
    std::shared_ptr<Worker>& worker;
    TaskGraph& graph;
    TaskChunk chunk;
    MemoryId memory_id;
    std::vector<BufferRequirement> buffers;
    EventList dependencies;
};

struct TaskResult {
    std::shared_ptr<Worker>& worker;
    TaskGraph& graph;
    EventList events;
};

}  // namespace kmm
#pragma once

#include "kmm/core/buffer.hpp"
#include "kmm/core/identifiers.hpp"

namespace kmm {

class TaskGraph;
class Worker;

struct TaskBuilder {
    TaskGraph& graph;
    std::shared_ptr<Worker>& worker;
    MemoryId memory_id;
    std::vector<BufferRequirement> buffers;
    EventList dependencies;
};

struct TaskResult {
    TaskGraph& graph;
    std::shared_ptr<Worker>& worker;
    EventList events;
};

}  // namespace kmm
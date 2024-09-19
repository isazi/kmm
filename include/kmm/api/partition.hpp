#pragma once

#include <cstddef>
#include <vector>

#include "kmm/core/geometry.hpp"
#include "kmm/core/identifiers.hpp"
#include "kmm/core/work_chunk.hpp"
#include "kmm/internals/worker.hpp"

namespace kmm {

struct Chunk {
    ProcessorId owner_id;
    WorkIndex offset;
    WorkDim size;
};

struct Partition {
    std::vector<Chunk> chunks;
};

struct ChunkPartitioner {
    ChunkPartitioner(WorkDim chunk_size) : m_chunk_size(chunk_size) {}
    ChunkPartitioner(
        int64_t x,
        int64_t y = std::numeric_limits<int64_t>::max(),
        int64_t z = std::numeric_limits<int64_t>::max()) :
        m_chunk_size(x, y, z) {}

    Partition operator()(WorkDim index_space, const SystemInfo& info) const;

  private:
    WorkDim m_chunk_size;
};

}  // namespace kmm
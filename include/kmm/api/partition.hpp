#pragma once

#include <cstddef>
#include <vector>

#include "kmm/core/geometry.hpp"
#include "kmm/core/identifiers.hpp"
#include "kmm/internals/worker.hpp"

namespace kmm {

template<size_t N>
struct Chunk {
    ProcessorId owner_id;
    point<N> offset;
    dim<N> size;
};

template<size_t N>
struct Partition {
    std::vector<Chunk<N>> chunks;
};

template<size_t N>
struct ChunkPartitioner {
    dim<N> chunk_size;

    Partition<N> operator()(rect<N> index_space, const SystemInfo& info) const;
};

}  // namespace kmm
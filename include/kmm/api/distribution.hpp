#pragma once

#include <vector>

#include "runtime_impl.hpp"

#include "kmm/core/geometry.hpp"

namespace kmm {

template<size_t N>
struct Chunk {
    dim<N> global_size;
    rect<N> local_size;
    DeviceId owner_id;
};

template<size_t N>
struct Partition {
    dim<N> global_size;
    std::vector<Chunk<N>> chunks;

    Partition<N> operator()(RuntimeImpl* runtime) const {
        return *this;
    }
};

template<size_t N>
struct ChunkPartition {
    dim<N> global_size;
    dim<N> chunk_size;

    ChunkPartition(dim<N> global_size, dim<N> chunk_size) :
        global_size(global_size),
        chunk_size(chunk_size) {}

    Partition<N> operator()(RuntimeImpl* runtime) const;
};

}  // namespace kmm
#include "spdlog/spdlog.h"

#include "kmm/api/partition.hpp"
#include "kmm/utils/integer_fun.hpp"

namespace kmm {

template<size_t N>
Partition<N> ChunkPartitioner<N>::operator()(rect<N> index_space, const SystemInfo& info) const {
    std::vector<Chunk<N>> chunks;
    size_t num_devices = info.num_devices();

    if (num_devices == 0) {
        throw std::runtime_error("no CUDA devices found, cannot partition work");
    }

    dim<N> num_chunks;
    point<N> current;
    size_t num_total_chunks = 1;

    for (size_t i = 0; i < N; i++) {
        num_chunks[i] = div_ceil(index_space.sizes[i], chunk_size[i]);
        num_total_chunks *= checked_cast<size_t>(num_chunks[i]);
    }

    size_t owner_id = 0;

    for (size_t it = 0; it < num_total_chunks; it++) {
        point<N> offset;
        dim<N> size;

        for (size_t i = 0; i < N; i++) {
            offset[i] = current[i] * chunk_size[i] + index_space.offset[i];
            size[i] = std::min(chunk_size[i], index_space.sizes[i] - offset[i]);
        }

        chunks.push_back(Chunk<N> {DeviceId(owner_id), offset, size});

        owner_id = (owner_id + 1) % num_devices;

        for (size_t i = 0; i < N; i++) {
            current[i] += 1;

            if (current[i] >= num_chunks[i]) {
                current[i] = 0;
            } else {
                break;
            }
        }
    }

    return {std::move(chunks)};
}

template struct ChunkPartitioner<0>;
template struct ChunkPartitioner<1>;
template struct ChunkPartitioner<2>;
template struct ChunkPartitioner<3>;
template struct ChunkPartitioner<4>;
template struct ChunkPartitioner<5>;
template struct ChunkPartitioner<6>;

}  // namespace kmm
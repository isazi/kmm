#include "kmm/api/partition.hpp"
#include "kmm/utils/integer_fun.hpp"

namespace kmm {

template<size_t N>
Partition<N> ChunkPartition<N>::operator()(const SystemInfo& info) const {
    std::vector<Chunk<N>> chunks;
    size_t num_devices = info.num_devices();

    dim<N> num_chunks;
    point<N> current;
    size_t num_total_chunks = 1;

    for (size_t i = 0; i < N; i++) {
        num_chunks[i] = div_ceil(global_size[i], chunk_size[i]);
        num_total_chunks *= checked_cast<size_t>(num_chunks[i]);
    }

    size_t owner_id = 0;

    for (size_t it = 0; it < num_total_chunks; it++) {
        point<N> offset;
        dim<N> size;

        for (size_t i = 0; i < N; i++) {
            offset[i] = current[i] * chunk_size[i];
            size[i] = std::min(chunk_size[i], global_size[i] - offset[i]);
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

template struct ChunkPartition<0>;
template struct ChunkPartition<1>;
template struct ChunkPartition<2>;
template struct ChunkPartition<3>;
template struct ChunkPartition<4>;
template struct ChunkPartition<5>;
template struct ChunkPartition<6>;

}  // namespace kmm
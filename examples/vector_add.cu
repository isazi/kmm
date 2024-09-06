
#include "kmm/api/access.hpp"
#include "kmm/api/runtime.hpp"
#include "kmm/api/array.hpp"
#include "kmm/api/partition.hpp"

void fill_range(
    kmm::rect<1> subrange,
    kmm::subview_mut<float> view
) {
    for (long i = subrange.begin(); i < subrange.end(); i++) {
        view[i] = float(i);
    }
}

void cuda_fill_zeros(
    kmm::CudaDevice& device,
    kmm::rect<1> subrange,
    kmm::cuda_subview_mut<float> view
) {
    device.fill(
        view.data_at(subrange.begin()),
        subrange.size(),
        0.0F
   );
}

__global__ void vector_add(
    kmm::rect<1> subrange,
    kmm::cuda_subview_mut<float> C,
    kmm::cuda_subview<float> A,
    kmm::cuda_subview<float> B
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x + subrange.begin();
    if (x >= subrange.end()) {
        return;
    }

    C[x] = A[x] + B[x];
}

int main() {
    using namespace kmm::placeholders;
    auto rt = kmm::make_runtime();

    int n = 1024 * 1024;
    int chunk_size = n / 10;

    auto A = kmm::Array<float>(n);
    auto B = kmm::Array<float>(n);
    auto C = kmm::Array<float>(n);

    rt.parallel_submit(
        kmm::ChunkPartition<1>{n, chunk_size},
        kmm::Host(),
        fill_range,
        write(A, slice(_x))
    );

    rt.parallel_submit(
        kmm::ChunkPartition<1>{n, chunk_size},
        kmm::Cuda(),
        cuda_fill_zeros,
        write(B, slice(_x))
    );

    rt.parallel_submit(
        kmm::ChunkPartition<1>{n, chunk_size},
        kmm::CudaKernel(256),
        vector_add,
        write(C, slice(_x)),
        read(A, slice(_x)),
        read(B, slice(_x))
    );

    rt.synchronize();
}
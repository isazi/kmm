
#include "kmm/api/access.hpp"
#include "kmm/api/runtime.hpp"
#include "kmm/api/array.hpp"
#include "kmm/api/partition.hpp"

__global__
void process_kernel(
    kmm::rect<3> subregion,
    int n, int m, int k,
    kmm::cuda_subview<float, 2> a,
    kmm::cuda_subview<float, 2> b,
    kmm::cuda_subview_mut<float, 2> c
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + subregion.offset.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y + subregion.offset.y;

    if (!subregion.contains(kmm::point {i, j, 0})) {
        return;
    }

    for (int l = 0; l < k; l++) {
        c[i][j] += a[i][l] * b[l][j];
    }
}

void kernel_function(
    kmm::rect<3> subregion,
    int n, int m, int k,
    kmm::subview<float, 2> a,
    kmm::subview<float, 2> b,
    kmm::subview_mut<float, 2> c
) {
    for (int i = subregion.begin(0); i < subregion.end(0); i++) {
        for (int j = subregion.begin(1); j < subregion.end(1); j++) {
            for (int l = 0; l < k; l++) {
                c[i][j] += a[i][l] * b[l][j];
            }
        }
    }
}

void example(kmm::Runtime rt) {
    using namespace kmm::placeholders;

    int n = 150;
    int m = 50;
    int k = 10;

    auto a = kmm::Array<float, 2> {n, k};
    auto b = kmm::Array<float, 2> {k, m};
    auto c = kmm::Array<float, 2> {n, m};

    rt.parallel_submit(
        kmm::ChunkPartition<3> {
            {n, m, k},
            {10, 10, k}
        },
        kmm::CudaKernel(
            dim3 {5, 5, 5}
        ),
        process_kernel,
        n,
        m,
        k,
        read(a, slice(_i, _k)),
        read(b, slice(_k, _j)),
        write(c, slice(_i, _j))
    );

    rt.synchronize();
}

int main() {
    auto rt = kmm::make_runtime();
    example(rt);
}
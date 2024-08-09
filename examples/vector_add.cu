#include "kmm/api/runtime.hpp"
#include "kmm/api/array.hpp"

__global__
void process(
    kmm::rect<3> subregion,
    int n, int m, int k,
    kmm::subview<float, 2> a,
    kmm::subview<float, 2> b,
    kmm::subview_mut<float, 2> c
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x + subregion.offset.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y + subregion.offset.y;

    if (subregion.contains(kmm::point {x, y, 0})) {
        
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

    kmm::Array<float, 2> a = {n, k};
    kmm::Array<float, 2> b = {k, m};
    kmm::Array<float, 2> c = {n, m};

    rt.parallel_submit(
        kmm::ChunkPartition<3> {
            {n, m, k},
            {10, 10, k}
        },
        kmm::CudaKernel(
            {5, 5, 5}
        ),
        kernel_function,
        n,
        m,
        k,
        read(a, slice(_x, _z)),
        read(b, slice(_z, _y)),
        write(c, slice(_x, _y))
    );

    rt.synchronize();
}

int main() {
    auto rt = kmm::make_runtime();
    example(rt);
}
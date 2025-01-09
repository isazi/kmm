#include <curand_kernel.h>

#include "kmm/api/mapper.hpp"
#include "kmm/api/runtime.hpp"

__global__ void cn_pnpoly(
    kmm::NDRange chunk,
    kmm::gpu_subview_mut<int> bitmap,
    kmm::gpu_subview<float2> points,
    int nvertices,
    kmm::gpu_view<float2> vertices
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + chunk.begin(0);

    if (i < chunk.end(0)) {
        int c = 0;
        float2 p = points[i];

        int k = nvertices - 1;

        for (int j = 0; j < nvertices; k = j++) {  // edge from v to vp
            float2 vj = vertices[j];
            float2 vk = vertices[k];

            float slope = (vk.x - vj.x) / (vk.y - vj.y);

            if (((vj.y > p.y) != (vk.y > p.y)) &&  //if p is between vj and vk vertically
                (p.x < slope * (p.y - vj.y) + vj.x
                )) {  //if p.x crosses the line vj-vk when moved in positive x-direction
                c = !c;
            }
        }

        bitmap[i] = c;  // 0 if even (out), and 1 if odd (in)
    }
}

__global__ void init_points(kmm::NDRange chunk, kmm::gpu_subview_mut<float2> points) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + chunk.begin(0);

    if (i < chunk.end(0)) {
        curandStatePhilox4_32_10_t state;
        curand_init(1234, i, 0, &state);
        points[i] = {curand_normal(&state), curand_normal(&state)};
    }
}

void init_polygon(kmm::NDRange chunk, int nvertices, kmm::view_mut<float2> vertices) {
    for (int64_t i = chunk.begin(); i < chunk.end(); i++) {
        float angle = float(i) / float(nvertices) * float(2.0F * M_PI);
        vertices[i] = {cosf(angle), sinf(angle)};
    }
}

int main() {
    using namespace kmm::placeholders;
    spdlog::set_level(spdlog::level::trace);

    auto rt = kmm::make_runtime();
    int nvertices = 1000;
    int npoints = 1'000'000'000;
    int npoints_per_chunk = npoints / 10;
    dim3 block_size = 256;

    auto vertices = kmm::Array<float2> {nvertices};
    auto points = kmm::Array<float2> {npoints};
    auto bitmap = kmm::Array<int> {npoints};

    rt.submit(
        {nvertices},
        kmm::ProcessorId::host(),
        kmm::Host(init_polygon),
        nvertices,
        write(vertices)
    );

    rt.parallel_submit(
        {npoints},
        {npoints_per_chunk},
        kmm::GPUKernel(init_points, block_size),
        write(points(_x))
    );

    rt.parallel_submit(
        {npoints},
        {npoints_per_chunk},
        kmm::GPUKernel(cn_pnpoly, block_size),
        write(bitmap(_x)),
        points(_x),
        nvertices,
        vertices
    );

    rt.synchronize();

    return EXIT_SUCCESS;
}

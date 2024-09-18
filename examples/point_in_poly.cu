#include <curand_kernel.h>

#include "kmm/api/access.hpp"
#include "kmm/api/runtime.hpp"

__global__ void cn_pnpoly(
    kmm::rect<1> subrange,
    kmm::cuda_subview_mut<int> bitmap,
    kmm::cuda_subview<float2> points,
    int nvertices,
    kmm::cuda_view<float2> vertices
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + subrange.begin();

    if (i < subrange.end()) {
        int c = 0;
        float2 p = points[i];

        int k = nvertices-1;

        for (int j=0; j<nvertices; k = j++) {    // edge from v to vp
            float2 vj = vertices[j];
            float2 vk = vertices[k];

            float slope = (vk.x-vj.x) / (vk.y-vj.y);

            if ( (  (vj.y>p.y) != (vk.y>p.y)) &&            //if p is between vj and vk vertically
                (p.x < slope * (p.y-vj.y) + vj.x) ) {   //if p.x crosses the line vj-vk when moved in positive x-direction
                c = !c;
            }
        }

        bitmap[i] = c; // 0 if even (out), and 1 if odd (in)
    }
}

__global__ void init_points(
    kmm::rect<1> subrange,
    kmm::cuda_subview_mut<float2> points
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + subrange.begin();

    if (i < subrange.end()) {
        curandStatePhilox4_32_10_t state;
        curand_init(1234, i, 0, &state);
        points[i] = {curand_normal(&state), curand_normal(&state)};
    }
}

void init_polygon(
    kmm::rect<1> subrange,
    int nvertices,
    kmm::view_mut<float2> vertices
) {
    for (int64_t i = subrange.begin(); i < subrange.end(); i++) {
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

    rt.parallel_for(
        {npoints},
        {npoints_per_chunk},
        kmm::CudaKernel(init_points, block_size),
        write(points, slice(_x))
    );

    rt.parallel_for(
        {npoints},
        {npoints_per_chunk},
        kmm::CudaKernel(cn_pnpoly, block_size),
        write(bitmap, slice(_x)),
        read(points, slice(_x)),
        nvertices,
        read(vertices)
    );

    rt.synchronize();
}
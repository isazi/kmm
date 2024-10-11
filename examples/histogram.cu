#include <random>

#include "kmm/kmm.hpp"

void initialize_image(
    unsigned long seed,
    int width,
    int height,
    kmm::subview_mut<uint8_t, 2> image
) {
    std::default_random_engine rand {seed};
    std::uniform_int_distribution<uint8_t> dist {};

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            image[i][j] = dist(rand);
        }
    }
}

void initialize_images(
    kmm::WorkRange subrange,
    int width,
    int height,
    kmm::subview_mut<uint8_t, 3> images
) {
    for (int i = subrange.begin(); i < subrange.end(); i++) {
        initialize_image(i, width, height, images.drop_axis<0>(i));
    }
}

__global__ void calculate_histogram(
    kmm::WorkRange subrange,
    int width,
    int height,
    kmm::cuda_subview<uint8_t, 3> images,
    kmm::cuda_view_mut<int> histogram
) {
    int index = blockIdx.z * blockDim.z + threadIdx.z + subrange.begin.z;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < subrange.end.z && i < height && j < width) {
        uint8_t value = images[index][i][j];
        atomicAdd(&histogram[value], 1);
    }
}

int main() {
    using namespace kmm::placeholders;

    auto rt = kmm::make_runtime();
    int width = 1080;
    int height = 1920;
    int num_images = 10'000;
    int images_per_chunk = 500;
    dim3 block_size = 256;

    auto images = kmm::Array<uint8_t, 3>{{num_images, height, width}};
    auto histogram = kmm::Array<int> {256};

    rt.parallel_submit(
        {num_images},
        {images_per_chunk},
        kmm::Host(initialize_images),
        width,
        height,
        write(images, slice(_x, _, _))
    );

    images.synchronize();

    rt.parallel_submit(
        {width, height, num_images},
        {width, height, images_per_chunk},
        kmm::CudaKernel(calculate_histogram, block_size),
        width,
        height,
        read(images, slice(_z, _y, _x)),
        reduce(histogram, kmm::ReductionOp::Sum)
    );

    rt.synchronize();

    return 0;
}
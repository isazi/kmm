#include <random>

#include "kmm/kmm.hpp"

void initialize_image(
    unsigned long seed,
    int width,
    int height,
    kmm::subview_mut<uint8_t, 2> image
) {
    std::mt19937 rand {seed};
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
    for (auto i = int(subrange.begin()); i < subrange.end(); i++) {
        initialize_image(i, width, height, images.drop_axis<0>(i));
    }
}

__global__ void calculate_histogram(
    kmm::WorkRange subrange,
    int width,
    int height,
    kmm::cuda_subview<uint8_t, 3> images,
    kmm::cuda_subview_mut<int, 2> histogram
) {
    int image_id = blockIdx.z * blockDim.z + threadIdx.z + subrange.begin.z;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (image_id < int(subrange.end.z) && i < height && j < width) {
        uint8_t value = images[image_id][i][j];
        atomicAdd(&histogram[image_id][value], 1);
    }
}

int main() {
    using namespace kmm::placeholders;
    spdlog::set_level(spdlog::level::trace);

    auto rt = kmm::make_runtime();
    int width = 1080;
    int height = 1920;
    int num_images = 2500;
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
        write(images, access(_x, _, _))
    );

    rt.synchronize();

    rt.parallel_submit(
        {width, height, num_images},
        {width, height, images_per_chunk},
        kmm::CudaKernel(calculate_histogram, block_size),
        width,
        height,
        read(images, access(_z, _y, _x)),
        reduce(histogram, kmm::ReductionOp::Sum, privatize(_z), access(_))
    );

    rt.synchronize();

    return 0;
}
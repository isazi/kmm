
#include "spdlog/spdlog.h"
#include "kmm/api/access.hpp"
#include "kmm/api/array.hpp"
#include "kmm/api/partition.hpp"
#include "kmm/api/runtime.hpp"

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
    spdlog::debug("cuda_fill_zeros: {} {} {}", (void*)view.data(), (void*)view.data_at(subrange.begin()), subrange.begin());

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
    spdlog::set_level(spdlog::level::trace);

    auto rt = kmm::make_runtime();

    for (size_t repeat = 2; repeat > 0; repeat--) {
        int64_t n = (5L * 1000 * 1000 * 1000) / 4;
        int64_t chunk_size = n / 20;

        auto A = kmm::Array<float>(n);
        auto B = kmm::Array<float>(n);
        auto C = kmm::Array<float>(n);

        rt.parallel_for(
            {n},
            {chunk_size},
            kmm::Host(fill_range),
            write(A, slice(_x))
        );

        rt.parallel_for(
            {n},
            {chunk_size},
            kmm::Cuda(cuda_fill_zeros),
            write(B, slice(_x))
        );

        for (size_t x = 0; x < 10; x++) {
            C = kmm::Array<float>(n);

            rt.parallel_for(
                {n},
                {chunk_size},
                kmm::CudaKernel(vector_add, 256),
                write(C, slice(_x)),
                read(A, slice(_x)),
                read(B, slice(_x)));
        }

        rt.synchronize();
    }
}

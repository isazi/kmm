#include "spdlog/spdlog.h"

#include "kmm/kmm.hpp"

#include "vector_add.h"
#include "vector_add.cuh"


int main(void) {
    unsigned int threads_per_block = 256;
    unsigned int n_blocks = ceil((1.0 * SIZE) / threads_per_block);
    int n = SIZE;

    // Create manager
    auto manager = kmm::build_runtime();
    spdlog::set_level(spdlog::level::debug);

    // Request 3 memory areas of a certain size
    auto A = kmm::Array<float>(n);
    auto B = kmm::Array<float>(n);
    auto C = kmm::Array<float>(n);

    // Initialize array A and B on the host
    manager.submit(kmm::Host(), initialize, write(A), write(B));

    // Execute the function on the device.
    manager.submit(kmm::CudaKernel(n_blocks, threads_per_block), vector_add, A, B, write(C), n);

    // Verify the result on the host
    auto verify_event = manager.submit(kmm::Host(), verify, C);

    manager.synchronize();
    std::cout << "done\n";

    return 0;
}

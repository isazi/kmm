#include <iostream>

#include "spdlog/spdlog.h"

#include "kmm/platforms/host.hpp"
#include "kmm/runtime.hpp"

#define SIZE 65536

// __global__ void vector_add(float* A, float* B, float* C, unsigned int size) {
//     unsigned int item = (blockDim.x * blockIdx.x) + threadIdx.x;

//     if (item < size) {
//         C[item] = A[item] + B[item];
//     }
// }

void initialize(float* A, float* B, unsigned int size) {
    for (unsigned int item = 0; item < size; item++) {
        reinterpret_cast<float*>(A)[item] = 1.0;
        reinterpret_cast<float*>(B)[item] = 2.0;
    }
}

void execute(float* C, unsigned int size) {}

void verify(const float* C, unsigned int size) {
    for (unsigned int item = 0; item < size; item++) {
        if ((C[item] - 3.0) > 1.0e-9) {
            std::cout << "ERROR" << std::endl;
            break;
        }
    }
    std::cout << "SUCCESS" << std::endl;
}

int main(void) {
    spdlog::set_level(spdlog::level::trace);

    unsigned int threads_per_block = 1024;
    unsigned int n_blocks = ceil((1.0 * SIZE) / threads_per_block);
    int n = SIZE;

    // Create memory manager
    auto manager = kmm::build_runtime();

    // Request 3 memory areas of a certain size
    auto A = manager.allocate<float>({n});
    auto B = manager.allocate<float>({n});
    auto C = manager.allocate<float>({n});

    // Create devices
    //    auto cpu = kmm::CPU();
    //    auto gpu = kmm::CUDA();

    // TODO: run initialization

    // Copy data to the GPU
    //    manager.move_to(gpu, A);
    //    manager.move_to(gpu, B);

    // TODO: run kernel
    manager.submit(kmm::Host(), execute, write(C), 100);

    //    manager.copy_release(C, C_h);
    // TODO: run verify
    manager.submit(kmm::Host(), verify, C, 100);

    manager.barrier().wait();
    return 0;
}

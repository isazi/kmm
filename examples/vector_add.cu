#include <iostream>

#include "spdlog/spdlog.h"

#include "kmm/array.hpp"
#include "kmm/cuda/cuda.hpp"
#include "kmm/future.hpp"
#include "kmm/host/host.hpp"
#include "kmm/runtime.hpp"

#define SIZE 6553600

// __global__ void vector_add(float* A, float* B, float* C, unsigned int size) {
//     unsigned int item = (blockDim.x * blockIdx.x) + threadIdx.x;

//     if (item < size) {
//         C[item] = A[item] + B[item];
//     }
// }

void initialize(float* A, float* B) {
    for (unsigned int item = 0; item < SIZE; item++) {
        reinterpret_cast<float*>(A)[item] = 1.0;
        reinterpret_cast<float*>(B)[item] = 2.0;
    }
    //        std::cout << "initialize\n";
}

__global__ void execute_kernel(float* C, const float* A, const float* B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < SIZE) {
        C[i] = A[i] + B[i];
    }
}

void execute(kmm::CudaExecutor& executor, float* C, const float* A, const float* B) {
    int block_size = 256;
    int num_blocks = (SIZE + block_size - 1) / block_size;
    execute_kernel<<<num_blocks, block_size, 0, executor.stream()>>>(C, A, B);
}

void verify(const float* C) {
    for (unsigned int item = 0; item < SIZE; item++) {
        if ((C[item] - 3.0) > 1.0e-9) {
            std::cout << "ERROR" << std::endl;
            break;
        }
    }

    std::cout << "SUCCESS" << std::endl;
}

int main(void) {
    spdlog::set_level(spdlog::level::debug);

    unsigned int threads_per_block = 1024 * 1024;
    unsigned int n_blocks = ceil((1.0 * SIZE) / threads_per_block);
    int n = SIZE;

    // Create manager
    auto manager = kmm::build_runtime();

    for (size_t i = 0; i < 100000; i++) {
        // Request 3 memory areas of a certain size
        auto A = kmm::Array<float>(n);
        auto B = kmm::Array<float>(n);
        auto C = kmm::Array<float>(n);
        auto size = kmm::Future<unsigned int>();

        // Initialize array A and B on the host
        manager.submit(kmm::Host(), initialize, write(A), write(B));

        // Execute the function on the device.
        if (i % 2 == 0) {
            manager.submit(kmm::Cuda(), execute, write(C), A, B);
        } else {
            int block_size = 256;
            int num_blocks = (SIZE + block_size - 1) / block_size;

            manager.submit(kmm::CudaKernel(num_blocks, block_size), execute_kernel, write(C), A, B);
        }

        // Verify the result on the host.
        auto verify_id = manager.submit(kmm::Host(), verify, C);

        manager.synchronize();
    }

    manager.synchronize();
    std::cout << "done\n";

    return 0;
}

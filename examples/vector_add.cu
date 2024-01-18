#include <deque>
#include <iostream>

#include "spdlog/spdlog.h"

#include "kmm/array.hpp"
#include "kmm/cuda/cuda.hpp"
#include "kmm/future.hpp"
#include "kmm/host/host.hpp"
#include "kmm/runtime.hpp"

#define SIZE 65536000

__global__ void vector_add(const float* A, const float* B, float* C, unsigned int size) {
    unsigned int item = (blockDim.x * blockIdx.x) + threadIdx.x;

    if (item < size) {
        C[item] = A[item] + B[item];
    }
}

void initialize(float* A, float* B) {
#pragma omp parallel for
    for (unsigned int item = 0; item < SIZE; item++) {
        reinterpret_cast<float*>(A)[item] = 1.0;
        reinterpret_cast<float*>(B)[item] = 2.0;
    }

    std::cout << "initialize\n";
}

void execute(kmm::CudaExecutor& executor, float* C, const float* A, const float* B) {
    int block_size = 256;
    int num_blocks = (SIZE + block_size - 1) / block_size;
    vector_add<<<num_blocks, block_size, 0, executor.stream()>>>(A, B, C, SIZE);

    executor.launch(num_blocks, block_size, 0, vector_add, A, B, C, SIZE);
}

void verify(const float* C) {
#pragma omp parallel for
    for (unsigned int item = 0; item < SIZE; item++) {
        if (fabsf(C[item] - 3.0f) > 1.0e-9) {
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
    std::deque<kmm::EventId> events;

    kmm::Array<int> x = manager.allocate({1, 2, 3});
    spdlog::warn("x: {}\n", x.read()[0]);

    for (size_t i = 0; i < 20; i++) {
        // Request 3 memory areas of a certain size
        auto A = kmm::Array<float>(n);
        auto B = kmm::Array<float>(n);
        auto C = kmm::Array<float>(n);
        auto size = kmm::Future<unsigned int>();

        // Initialize array A and B on the host
        manager.submit(kmm::Host(), initialize, write(A), write(B));

        // Execute the function on the device.
        manager.submit(kmm::Cuda(), execute, write(C), A, B);
        manager.submit(kmm::CudaKernel(grid_dim, block_dim), vector_add, write(C), A, B);

        // Verify the result on the host.
        auto verify_id = manager.submit(kmm::Host(), verify, C);
        events.push_back(verify_id);

        while (events.size() > 10) {
            manager.wait(events.front());
            events.pop_front();
        }
        //        manager.synchronize();
    }

    manager.synchronize();
    std::cout << "done\n";

    return 0;
}

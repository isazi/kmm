#include <deque>
#include <iostream>

#include "spdlog/spdlog.h"

#include "kmm/array.hpp"
#include "kmm/cuda/cuda.hpp"
#include "kmm/host/host.hpp"
#include "kmm/runtime.hpp"

#define SIZE 65536000

__global__ void vector_add(const float* A, const float* B, float* C, int size) {
    int item = (blockDim.x * blockIdx.x) + threadIdx.x;

    if (item < size) {
        C[item] = A[item] + B[item];
    }
}

void initialize(float* A, float* B) {
#pragma omp parallel for
    for (unsigned int item = 0; item < SIZE; item++) {
        A[item] = 1.0;
        B[item] = 2.0;
    }

    std::cout << "initialize\n";
}

void execute(kmm::CudaDevice& device, float* C, const float* A, const float* B) {
    int block_size = 256;
    int num_blocks = (SIZE + block_size - 1) / block_size;
    vector_add<<<num_blocks, block_size, 0, device.stream()>>>(A, B, C, SIZE);
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

    unsigned int threads_per_block = 256;
    unsigned int n_blocks = ceil((1.0 * SIZE) / threads_per_block);
    int n = SIZE;

    // Create manager
    auto manager = kmm::build_runtime();
    std::deque<kmm::EventId> events;

    for (size_t i = 0; i < 20; i++) {
        // Request 3 memory areas of a certain size
        auto A = kmm::Array<float>(n);
        auto B = kmm::Array<float>(n);
        auto C = kmm::Array<float>(n);

        // Initialize array A and B on the host
        manager.submit(kmm::Host(), initialize, write(A), write(B));

        // Execute the function on the device.
        //manager.submit(kmm::Cuda(), execute, write(C), A, B);
        manager.submit(kmm::CudaKernel(n_blocks, threads_per_block), vector_add, A, B, write(C), n);

        // Verify the result on the host.
        auto verify_event = manager.submit(kmm::Host(), verify, C);

        events.push_back(verify_event);
        if (events.size() >= 5) {
            manager.wait(events[events.size() - 5]);
        }
    }

    manager.synchronize();
    std::cout << "done\n";

    return 0;
}

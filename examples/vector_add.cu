#include <iostream>

#include "spdlog/spdlog.h"

#include "kmm/array.hpp"
#include "kmm/cuda/cuda.hpp"
#include "kmm/host/executor.hpp"
#include "kmm/runtime.hpp"
#include "kmm/scalar.hpp"

#define SIZE 6553600

// __global__ void vector_add(float* A, float* B, float* C, unsigned int size) {
//     unsigned int item = (blockDim.x * blockIdx.x) + threadIdx.x;

//     if (item < size) {
//         C[item] = A[item] + B[item];
//     }
// }

void initialize(float* A, float* B, unsigned int& size) {
    size = SIZE;
    printf("A %p %d\n", &size, size);

    for (unsigned int item = 0; item < size; item++) {
        reinterpret_cast<float*>(A)[item] = 1.0;
        reinterpret_cast<float*>(B)[item] = 2.0;
    }
    //        std::cout << "initialize\n";
}

void execute(float* C, const float* A, const float* B, const unsigned int& size) {
    printf("B %p %d\n", &size, size);
    for (unsigned int item = 0; item < size; item++) {
        C[item] = A[item] + B[item];
    }
    //        std::cout << "execute\n";
}

void verify(const float* C, const unsigned int& size) {
    printf("C %p %d\n", &size, size);
    for (unsigned int item = 0; item < size; item++) {
        if ((C[item] - 3.0) > 1.0e-9) {
            std::cout << "ERROR" << std::endl;
            break;
        }
    }
    //        std::cout << "SUCCESS" << std::endl;
}

int main(void) {
    spdlog::set_level(spdlog::level::debug);

    unsigned int threads_per_block = 1024 * 1024;
    unsigned int n_blocks = ceil((1.0 * SIZE) / threads_per_block);
    int n = SIZE;

    // Create manager
    auto manager = kmm::build_runtime();

    for (size_t i = 0; i < 100000; i++) {
        auto event = kmm::EventId(0);

        for (size_t j = 0; j < 2; j++) {
            // Request 3 memory areas of a certain size
            auto A = kmm::Array<float>(n);
            auto B = kmm::Array<float>(n);
            auto C = kmm::Array<float>(n);
            auto n_block = kmm::Scalar<unsigned int>();

            manager.submit(kmm::Host(), initialize, write(A), write(B), write(n_block));
            manager.submit(kmm::Cuda(), execute, write(C), A, B, n_block);
            manager.submit(kmm::Host(), verify, C, n_block);

            event = manager.join(event, manager.submit_barrier());
        }

        manager.wait(event);
    }

    manager.synchronize();
    std::cout << "done\n";

    return 0;
}

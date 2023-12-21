#include <iostream>

#include "spdlog/spdlog.h"

#include "kmm/array.hpp"
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
    //    std::cout << "initialize\n";
}

void execute(float* C, const float* A, const float* B, unsigned int size) {
    for (unsigned int item = 0; item < size; item++) {
        C[item] = A[item] + B[item];
    }
    //    std::cout << "execute\n";
}

void verify(const float* C, unsigned int size) {
    for (unsigned int item = 0; item < size; item++) {
        if ((C[item] - 3.0) > 1.0e-9) {
            std::cout << "ERROR" << std::endl;
            break;
        }
    }
    //    std::cout << "SUCCESS" << std::endl;
}

int main(void) {
    spdlog::set_level(spdlog::level::warn);

    unsigned int threads_per_block = 1024 * 1024;
    unsigned int n_blocks = ceil((1.0 * SIZE) / threads_per_block);
    int n = SIZE;

    // Create manager
    auto manager = kmm::build_runtime();

    for (size_t i = 0; i < 100000; i++) {
        // Request 3 memory areas of a certain size
        auto A = kmm::Array<float>({n});
        auto B = kmm::Array<float>({n});
        auto C = kmm::Array<float>({n});

        manager.submit(kmm::Host(), initialize, write(A), write(B), 100);
        manager.submit(kmm::Host(), execute, write(C), A, B, 100);
        manager.submit(kmm::Host(), verify, C, 100);
    }

    manager.synchronize();
    std::cout << "done\n";

    return 0;
}

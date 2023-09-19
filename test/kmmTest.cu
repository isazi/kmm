#include <gtest/gtest.h>

#include <type_traits>

#include "kmm.hpp"

TEST(GPU, ZeroInitialization) {
    auto gpu = kmm::GPU();
    EXPECT_EQ(gpu.device_id, 0);
}

TEST(GPU, Initialization) {
    auto gpu = kmm::GPU(2);
    EXPECT_EQ(gpu.device_id, 2);
}

TEST(GPU, Copy) {
    auto gpu = kmm::GPU(2);
    auto new_gpu = gpu;
    EXPECT_EQ(new_gpu.device_id, gpu.device_id);
}

TEST(Pointer, Initialized) {
    auto pointer = kmm::Pointer<kmm::Integer>(14);
    EXPECT_EQ(pointer.id, 14);
}

#include <gtest/gtest.h>
#include <type_traits>

#include "kmm.hpp"

TEST(GPU, ZeroInitialization) {
    auto gpu = kmm::GPU();
    EXPECT_EQ(gpu.device_id, 0);
}

TEST(GPU, Initialization) {
    auto gpu = KMM::GPU(2);
    EXPECT_EQ(gpu.device_id, 2);
}

TEST(GPU, Copy) {
    auto gpu = KMM::GPU(2);
    auto new_gpu = gpu;
    EXPECT_EQ(new_gpu.device_id, gpu.device_id);
}

TEST(Pointer, Initialized) {
    auto pointer = kmm::Pointer(14);
    EXPECT_EQ(pointer.id, 14);
    auto uint_type = kmm::UInteger();
    auto new_pointer = kmm::Pointer(3, uint_type);
    EXPECT_EQ(pointer.id, 3);
    EXPECT_EQ(std::underlying_type_t<pointer.type>, kmm::UInteger);
}

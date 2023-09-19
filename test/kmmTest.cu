#include <gtest/gtest.h>

#include <typeinfo>

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
    EXPECT_TRUE(typeid(pointer.type).hash_code() == typeid(kmm::Integer).hash_code());
    EXPECT_FALSE(typeid(pointer.type).hash_code() == typeid(kmm::UInteger).hash_code());
}

TEST(Pointer, DirtyByte) {
    auto pointer = kmm::Pointer<kmm::FP_Double>(3);
    EXPECT_FALSE(pointer.dirty);
    auto new_pointer = kmm::WritePointer(pointer);
    EXPECT_EQ(new_pointer.id, pointer.id);
    EXPECT_TRUE(pointer.dirty);
}

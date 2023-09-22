#include <gtest/gtest.h>

#include <typeinfo>

#include "kmm.hpp"

TEST(Misc, is_cpu) {
    auto cpu = kmm::CPU();
    auto gpu = kmm::CUDA();
    EXPECT_TRUE(kmm::is_cpu(cpu));
    EXPECT_FALSE(kmm::is_cpu(gpu));
}

TEST(Misc, same_device) {
    auto cpu = kmm::CPU();
    auto gpu_zero = kmm::CUDA(0);
    auto gpu_one = kmm::CUDA(1);
    EXPECT_TRUE(kmm::same_device(cpu, cpu));
    EXPECT_FALSE(kmm::same_device(cpu, gpu_zero));
    EXPECT_FALSE(kmm::same_device(gpu_one, cpu));
    EXPECT_FALSE(kmm::same_device(gpu_one, gpu_zero));
    EXPECT_TRUE(kmm::same_device(gpu_zero, gpu_zero));
}

TEST(CUDA, ZeroInitialization) {
    auto gpu = kmm::CUDA();
    EXPECT_EQ(gpu.device_id, 0);
}

TEST(CUDA, Initialization) {
    auto gpu = kmm::CUDA(2);
    EXPECT_EQ(gpu.device_id, 2);
}

TEST(CUDA, Copy) {
    auto gpu = kmm::CUDA(2);
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

TEST(Buffer, ZeroInitialization) {
    auto buffer = kmm::Buffer();
    EXPECT_EQ(buffer.getSize(), 0);
    EXPECT_FALSE(buffer.is_allocated());
    EXPECT_TRUE(
        typeid(dynamic_cast<kmm::UnknownDevice*>(buffer.getDevice().get())).hash_code()
        == typeid(kmm::UnknownDevice*).hash_code());
    EXPECT_FALSE(kmm::on_cpu(buffer));
}

TEST(Buffer, CPU) {
    auto cpu = kmm::CPU();
    auto buffer = kmm::Buffer(cpu, 42);
    EXPECT_EQ(buffer.getSize(), 42);
    EXPECT_FALSE(buffer.is_allocated());
    EXPECT_TRUE(kmm::on_cpu(buffer));
    EXPECT_FALSE(kmm::on_cuda(buffer));
}

TEST(Buffer, CUDA) {
    auto gpu = kmm::CUDA();
    auto buffer = kmm::Buffer(gpu, 13);
    EXPECT_EQ(buffer.getSize(), 13);
    EXPECT_FALSE(buffer.is_allocated());
    EXPECT_TRUE(kmm::on_cuda(buffer));
    EXPECT_FALSE(kmm::on_cpu(buffer));
}

#include <gtest/gtest.h>

#include "kmm.hpp"

TEST(Stream, NullInitialized) {
    auto stream = kmm::Stream(kmm::DeviceType::CPU);
    EXPECT_EQ(stream.cudaGetStream(), nullptr);
}

TEST(Stream, StreamInitialized) {
    auto stream = kmm::Stream(kmm::DeviceType::CUDA);
    EXPECT_NE(stream.cudaGetStream(), nullptr);
}

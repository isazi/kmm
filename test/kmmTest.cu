#include <gtest/gtest.h>

#include "kmm.hpp"

TEST(GPU, DeviceZero) {
    auto gpu = kmm::GPU();
    EXPECT_EQ(gpu.device_id, 0);
}

#include <gtest/gtest.h>

#include "kmm.hpp"

TEST(MemoryManager, StreamInitialized) {
    auto manager = kmm::MemoryManager();
    EXPECT_EQ(manager.getStream(), nullptr);
}

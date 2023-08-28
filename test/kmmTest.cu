#include <gtest/gtest.h>

#include "kmm.hpp"

TEST(MemoryManager, StreamInitialized) {
    auto manager = kmm::MemoryManager();
    EXPECT_EQ(manager.getStream(), nullptr);
}

TEST(MemoryManager, StreamAllocated) {
    auto manager = kmm::MemoryManager();
    unsigned int allocation_id = 0;

    allocation_id = manager.allocate(32);
    EXPECT_NE(manager.getStream(), nullptr);
}

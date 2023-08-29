#include <gtest/gtest.h>

#include "kmm.hpp"

TEST(MemoryManager, StreamInitialized) {
    auto manager = kmm::MemoryManager();
    EXPECT_EQ(manager.getStream(), nullptr);
}

TEST(MemoryManager, StreamAllocated) {
    auto manager = kmm::MemoryManager();
    manager.allocate(32);
    EXPECT_NE(manager.getStream(), nullptr);
}

TEST(MemoryManager, Release) {
    auto manager = kmm::MemoryManager();
    auto id = manager.allocate(1024);
    manager.release(id);
    EXPECT_EQ(manager.getPointer(id), nullptr);
}

#include <gtest/gtest.h>

#include "kmm.hpp"

TEST(MemoryManager, StreamInitialized) {
    manager = kmm::MemoryManager();
    EXPECT_EQ(manager.getStream(), nullptr);
}

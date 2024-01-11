#include <gtest/gtest.h>

#ifdef USE_CUDA

#include "kmm/cuda/memory_pool.hpp"

using namespace kmm;

TEST(MemoryPool, basics) {
    MemoryPool pool;

    // No blocks, allocation fails
    ASSERT_EQ(pool.allocate_range(123, 1), nullptr);

    // allocate some bytes
    pool.insert_block(reinterpret_cast<void*>(0x1000), 0x400);
    ASSERT_EQ(pool.allocate_range(0x100, 1), reinterpret_cast<void*>(0x1000));
    ASSERT_EQ(pool.allocate_range(0x200, 1), reinterpret_cast<void*>(0x1100));
    ASSERT_EQ(pool.allocate_range(0x200, 1), nullptr);  // fails
    ASSERT_EQ(pool.allocate_range(0x100, 1), reinterpret_cast<void*>(0x1300));
    ASSERT_EQ(pool.allocate_range(0x001, 1), nullptr);  // fails

    // deallocate everything
    ASSERT_EQ(pool.deallocate_range(reinterpret_cast<void*>(0x1100)), 512);
    ASSERT_EQ(pool.deallocate_range(reinterpret_cast<void*>(0x1000)), 256);
    ASSERT_EQ(pool.deallocate_range(reinterpret_cast<void*>(0x1300)), 256);

    // remove block
    void* addr;
    size_t size;
    ASSERT_TRUE(pool.remove_empty_block(&addr, &size));
    ASSERT_EQ(addr, reinterpret_cast<void*>(0x1000));
    ASSERT_EQ(size, 1024);
}

TEST(MemoryPool, alignment) {
    MemoryPool pool;
    pool.insert_block(reinterpret_cast<void*>(0x1000), 0xF00);
    ASSERT_EQ(pool.allocate_range(0x101, 4), reinterpret_cast<void*>(0x1000));
    ASSERT_EQ(pool.allocate_range(0x201, 4), reinterpret_cast<void*>(0x1104));
    ASSERT_EQ(pool.allocate_range(0x301, 4), reinterpret_cast<void*>(0x1308));
    ASSERT_EQ(pool.allocate_range(0x401, 4), reinterpret_cast<void*>(0x160c));

    // This one should fail
    ASSERT_EQ(pool.allocate_range(0x501, 4), nullptr);

    // Make a gap
    ASSERT_EQ(pool.deallocate_range(reinterpret_cast<void*>(0x1308)), 0x301);

    // These fit between the first and second allocation
    ASSERT_EQ(pool.allocate_range(1, 1), reinterpret_cast<void*>(0x1101));
    ASSERT_EQ(pool.allocate_range(2, 2), reinterpret_cast<void*>(0x1102));

    // These fill up the gap
    ASSERT_EQ(pool.allocate_range(0x100, 4), reinterpret_cast<void*>(0x1308));
    ASSERT_EQ(pool.allocate_range(0x101, 4), reinterpret_cast<void*>(0x1408));
    ASSERT_EQ(pool.allocate_range(0x100, 4), reinterpret_cast<void*>(0x150c));

    // These should fill up the remaining space
    ASSERT_EQ(pool.allocate_range(0x100, 4), reinterpret_cast<void*>(0x1a10));
    ASSERT_EQ(pool.allocate_range(0x100, 4), reinterpret_cast<void*>(0x1b10));
    ASSERT_EQ(pool.allocate_range(0x100, 4), reinterpret_cast<void*>(0x1c10));
    ASSERT_EQ(pool.allocate_range(0x100, 4), reinterpret_cast<void*>(0x1d10));
    ASSERT_EQ(pool.allocate_range(0x100, 4), nullptr);

    void* addr;
    size_t size;
    ASSERT_FALSE(pool.remove_empty_block(&addr, &size));

    // deallocate in an arbitrary order
    ASSERT_EQ(pool.deallocate_range(reinterpret_cast<void*>(0x1000)), 0x101);
    ASSERT_EQ(pool.deallocate_range(reinterpret_cast<void*>(0x1104)), 0x201);
    ASSERT_EQ(pool.deallocate_range(reinterpret_cast<void*>(0x160c)), 0x401);

    ASSERT_EQ(pool.deallocate_range(reinterpret_cast<void*>(0x1101)), 1);
    ASSERT_EQ(pool.deallocate_range(reinterpret_cast<void*>(0x1102)), 2);

    ASSERT_EQ(pool.deallocate_range(reinterpret_cast<void*>(0x1308)), 0x100);
    ASSERT_EQ(pool.deallocate_range(reinterpret_cast<void*>(0x1408)), 0x101);
    ASSERT_EQ(pool.deallocate_range(reinterpret_cast<void*>(0x150c)), 0x100);

    ASSERT_EQ(pool.allocate_range(0x100, 4), reinterpret_cast<void*>(0x1000));
    ASSERT_EQ(pool.deallocate_range(reinterpret_cast<void*>(0x1000)), 0x100);

    ASSERT_EQ(pool.deallocate_range(reinterpret_cast<void*>(0x1c10)), 0x100);
    ASSERT_EQ(pool.deallocate_range(reinterpret_cast<void*>(0x1a10)), 0x100);
    ASSERT_EQ(pool.deallocate_range(reinterpret_cast<void*>(0x1b10)), 0x100);
    ASSERT_EQ(pool.deallocate_range(reinterpret_cast<void*>(0x1d10)), 0x100);

    ASSERT_TRUE(pool.remove_empty_block(&addr, &size));
    ASSERT_EQ(addr, reinterpret_cast<void*>(0x1000));
    ASSERT_EQ(size, 0xF00);
}

#endif // USE_CUDA
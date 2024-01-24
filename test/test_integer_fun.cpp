#include <gtest/gtest.h>

#include "kmm/utils/integer_fun.hpp"

using namespace kmm;

TEST(IntegerFuns, round_up_to_multiple) {
    ASSERT_EQ(round_up_to_multiple(uint32_t(5), uint32_t(2)), uint32_t(6));
    ASSERT_EQ(round_up_to_multiple(uint32_t(103), uint32_t(5)), uint32_t(105));
    ASSERT_EQ(round_up_to_multiple(uint32_t(0), uint32_t(5)), uint32_t(0));

    ASSERT_EQ(round_up_to_multiple(uint64_t(5), uint64_t(2)), uint64_t(6));
    ASSERT_EQ(round_up_to_multiple(uint64_t(103), uint64_t(5)), uint64_t(105));
    ASSERT_EQ(round_up_to_multiple(uint64_t(0), uint64_t(5)), uint64_t(0));

    ASSERT_EQ(round_up_to_multiple(int32_t(5), int32_t(2)), int32_t(6));
    ASSERT_EQ(round_up_to_multiple(int32_t(5), int32_t(-2)), int32_t(6));
    ASSERT_EQ(round_up_to_multiple(int32_t(-5), int32_t(2)), int32_t(-4));
    ASSERT_EQ(round_up_to_multiple(int32_t(-5), int32_t(-2)), int32_t(-4));
    ASSERT_EQ(round_up_to_multiple(int32_t(6), int32_t(2)), int32_t(6));
    ASSERT_EQ(round_up_to_multiple(int32_t(6), int32_t(-2)), int32_t(6));
    ASSERT_EQ(round_up_to_multiple(int32_t(-6), int32_t(2)), int32_t(-6));
    ASSERT_EQ(round_up_to_multiple(int32_t(-6), int32_t(-2)), int32_t(-6));
    ASSERT_EQ(round_up_to_multiple(int32_t(103), int32_t(5)), int32_t(105));
    ASSERT_EQ(round_up_to_multiple(int32_t(0), int32_t(5)), int32_t(0));

    ASSERT_EQ(round_up_to_multiple(int64_t(5), int64_t(2)), int64_t(6));
    ASSERT_EQ(round_up_to_multiple(int64_t(5), int64_t(-2)), int64_t(6));
    ASSERT_EQ(round_up_to_multiple(int64_t(-5), int64_t(2)), int64_t(-4));
    ASSERT_EQ(round_up_to_multiple(int64_t(-5), int64_t(-2)), int64_t(-4));
    ASSERT_EQ(round_up_to_multiple(int64_t(6), int64_t(2)), int64_t(6));
    ASSERT_EQ(round_up_to_multiple(int64_t(6), int64_t(-2)), int64_t(6));
    ASSERT_EQ(round_up_to_multiple(int64_t(-6), int64_t(2)), int64_t(-6));
    ASSERT_EQ(round_up_to_multiple(int64_t(-6), int64_t(-2)), int64_t(-6));
    ASSERT_EQ(round_up_to_multiple(int64_t(103), int64_t(5)), int64_t(105));
    ASSERT_EQ(round_up_to_multiple(int64_t(0), int64_t(5)), int64_t(0));
}

TEST(IntegerFuns, round_up_to_power_of_two) {
    ASSERT_EQ(round_up_to_power_of_two(uint32_t(0)), 1);
    ASSERT_EQ(round_up_to_power_of_two(uint32_t(5)), 8);
    ASSERT_EQ(round_up_to_power_of_two(uint32_t(1024)), 1024);
    ASSERT_EQ(round_up_to_power_of_two(uint32_t(10000)), 16384);
    ASSERT_EQ(round_up_to_power_of_two(std::numeric_limits<uint32_t>::max()), 0);

    ASSERT_EQ(round_up_to_power_of_two(uint64_t(0)), 1);
    ASSERT_EQ(round_up_to_power_of_two(uint64_t(5)), 8);
    ASSERT_EQ(round_up_to_power_of_two(uint64_t(1024)), 1024);
    ASSERT_EQ(round_up_to_power_of_two(uint64_t(10000)), 16384);
    ASSERT_EQ(
        round_up_to_power_of_two(std::numeric_limits<uint64_t>::max()),
        0);

    ASSERT_EQ(round_up_to_power_of_two(int32_t(0)), 1);
    ASSERT_EQ(round_up_to_power_of_two(int32_t(5)), 8);
    ASSERT_EQ(round_up_to_power_of_two(int32_t(-5)), 1);
    ASSERT_EQ(round_up_to_power_of_two(int32_t(1024)), 1024);
    ASSERT_EQ(round_up_to_power_of_two(int32_t(-1024)), 1);
    ASSERT_EQ(round_up_to_power_of_two(int32_t(10000)), 16384);
    ASSERT_EQ(round_up_to_power_of_two(int32_t(-10000)), 1);
    ASSERT_EQ(round_up_to_power_of_two(std::numeric_limits<int32_t>::min()), 1);
    ASSERT_EQ(round_up_to_power_of_two(std::numeric_limits<int32_t>::max()), 0);

    ASSERT_EQ(round_up_to_power_of_two(int64_t(0)), 1);
    ASSERT_EQ(round_up_to_power_of_two(int64_t(5)), 8);
    ASSERT_EQ(round_up_to_power_of_two(int64_t(-5)), 1);
    ASSERT_EQ(round_up_to_power_of_two(int64_t(1024)), 1024);
    ASSERT_EQ(round_up_to_power_of_two(int64_t(-1024)), 1);
    ASSERT_EQ(round_up_to_power_of_two(int64_t(10000)), 16384);
    ASSERT_EQ(round_up_to_power_of_two(int64_t(-10000)), 1);
    ASSERT_EQ(round_up_to_power_of_two(std::numeric_limits<int64_t>::min()), 1);
    ASSERT_EQ(round_up_to_power_of_two(std::numeric_limits<int64_t>::max()), 0);
}

TEST(IntegerFuns, div_ceil) {
    ASSERT_EQ(div_ceil(uint32_t(5), uint32_t(2)), uint32_t(3));
    ASSERT_EQ(div_ceil(uint32_t(103), uint32_t(5)), uint32_t(21));
    ASSERT_EQ(div_ceil(uint32_t(0), uint32_t(5)), uint32_t(0));

    ASSERT_EQ(div_ceil(uint64_t(5), uint64_t(2)), uint64_t(3));
    ASSERT_EQ(div_ceil(uint64_t(103), uint64_t(5)), uint64_t(21));
    ASSERT_EQ(div_ceil(uint64_t(0), uint64_t(5)), uint64_t(0));

    ASSERT_EQ(div_ceil(int32_t(5), int32_t(2)), int32_t(3));
    ASSERT_EQ(div_ceil(int32_t(5), int32_t(-2)), int32_t(3));
    ASSERT_EQ(div_ceil(int32_t(-5), int32_t(2)), int32_t(-2));
    ASSERT_EQ(div_ceil(int32_t(-5), int32_t(-2)), int32_t(-2));
    ASSERT_EQ(div_ceil(int32_t(6), int32_t(2)), int32_t(3));
    ASSERT_EQ(div_ceil(int32_t(6), int32_t(-2)), int32_t(3));
    ASSERT_EQ(div_ceil(int32_t(-6), int32_t(2)), int32_t(-3));
    ASSERT_EQ(div_ceil(int32_t(-6), int32_t(-2)), int32_t(-3));
    ASSERT_EQ(div_ceil(int32_t(103), int32_t(5)), int32_t(21));
    ASSERT_EQ(div_ceil(int32_t(0), int32_t(5)), int32_t(0));

    ASSERT_EQ(div_ceil(int64_t(5), int64_t(2)), int64_t(3));
    ASSERT_EQ(div_ceil(int64_t(5), int64_t(-2)), int64_t(3));
    ASSERT_EQ(div_ceil(int64_t(-5), int64_t(2)), int64_t(-2));
    ASSERT_EQ(div_ceil(int64_t(-5), int64_t(-2)), int64_t(-2));
    ASSERT_EQ(div_ceil(int64_t(6), int64_t(2)), int64_t(3));
    ASSERT_EQ(div_ceil(int64_t(6), int64_t(-2)), int64_t(3));
    ASSERT_EQ(div_ceil(int64_t(-6), int64_t(2)), int64_t(-3));
    ASSERT_EQ(div_ceil(int64_t(-6), int64_t(-2)), int64_t(-3));
    ASSERT_EQ(div_ceil(int64_t(103), int64_t(5)), int64_t(21));
    ASSERT_EQ(div_ceil(int64_t(0), int64_t(5)), int64_t(0));
}

TEST(IntegerFuns, is_power_of_two) {
    ASSERT_FALSE(is_power_of_two(uint32_t(0)));
    ASSERT_TRUE(is_power_of_two(uint32_t(1)));
    ASSERT_TRUE(is_power_of_two(uint32_t(1024)));
    ASSERT_FALSE(is_power_of_two(uint32_t(1050)));
    ASSERT_FALSE(is_power_of_two(uint32_t(std::numeric_limits<uint32_t>::max())));

    ASSERT_FALSE(is_power_of_two(int32_t(0)));
    ASSERT_TRUE(is_power_of_two(int32_t(1)));
    ASSERT_TRUE(is_power_of_two(int32_t(1024)));
    ASSERT_FALSE(is_power_of_two(int32_t(1050)));
    ASSERT_FALSE(is_power_of_two(int32_t(std::numeric_limits<int32_t>::max())));

    ASSERT_FALSE(is_power_of_two(uint64_t(0)));
    ASSERT_TRUE(is_power_of_two(uint64_t(1)));
    ASSERT_TRUE(is_power_of_two(uint64_t(1024)));
    ASSERT_FALSE(is_power_of_two(uint64_t(1050)));
    ASSERT_FALSE(is_power_of_two(uint64_t(std::numeric_limits<uint64_t>::max())));

    ASSERT_FALSE(is_power_of_two(int64_t(0)));
    ASSERT_TRUE(is_power_of_two(int64_t(1)));
    ASSERT_TRUE(is_power_of_two(int64_t(1024)));
    ASSERT_FALSE(is_power_of_two(int64_t(1050)));
    ASSERT_FALSE(is_power_of_two(int64_t(std::numeric_limits<int64_t>::max())));
}
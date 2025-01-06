#include <cmath>

#include "catch2/catch_all.hpp"

#include "kmm/utils/integer_fun.hpp"
#define CHECK_EQ(A, B) CHECK((A) == (B))

using namespace kmm;

TEST_CASE("integer_fun, round_up_to_multiple") {
    CHECK_EQ(round_up_to_multiple(uint32_t(5), uint32_t(2)), uint32_t(6));
    CHECK_EQ(round_up_to_multiple(uint32_t(103), uint32_t(5)), uint32_t(105));
    CHECK_EQ(round_up_to_multiple(uint32_t(0), uint32_t(5)), uint32_t(0));

    CHECK_EQ(round_up_to_multiple(uint64_t(5), uint64_t(2)), uint64_t(6));
    CHECK_EQ(round_up_to_multiple(uint64_t(103), uint64_t(5)), uint64_t(105));
    CHECK_EQ(round_up_to_multiple(uint64_t(0), uint64_t(5)), uint64_t(0));

    CHECK_EQ(round_up_to_multiple(int32_t(5), int32_t(2)), int32_t(6));
    CHECK_EQ(round_up_to_multiple(int32_t(5), int32_t(-2)), int32_t(6));
    CHECK_EQ(round_up_to_multiple(int32_t(-5), int32_t(2)), int32_t(-4));
    CHECK_EQ(round_up_to_multiple(int32_t(-5), int32_t(-2)), int32_t(-4));
    CHECK_EQ(round_up_to_multiple(int32_t(6), int32_t(2)), int32_t(6));
    CHECK_EQ(round_up_to_multiple(int32_t(6), int32_t(-2)), int32_t(6));
    CHECK_EQ(round_up_to_multiple(int32_t(-6), int32_t(2)), int32_t(-6));
    CHECK_EQ(round_up_to_multiple(int32_t(-6), int32_t(-2)), int32_t(-6));
    CHECK_EQ(round_up_to_multiple(int32_t(103), int32_t(5)), int32_t(105));
    CHECK_EQ(round_up_to_multiple(int32_t(0), int32_t(5)), int32_t(0));

    CHECK_EQ(round_up_to_multiple(int64_t(5), int64_t(2)), int64_t(6));
    CHECK_EQ(round_up_to_multiple(int64_t(5), int64_t(-2)), int64_t(6));
    CHECK_EQ(round_up_to_multiple(int64_t(-5), int64_t(2)), int64_t(-4));
    CHECK_EQ(round_up_to_multiple(int64_t(-5), int64_t(-2)), int64_t(-4));
    CHECK_EQ(round_up_to_multiple(int64_t(6), int64_t(2)), int64_t(6));
    CHECK_EQ(round_up_to_multiple(int64_t(6), int64_t(-2)), int64_t(6));
    CHECK_EQ(round_up_to_multiple(int64_t(-6), int64_t(2)), int64_t(-6));
    CHECK_EQ(round_up_to_multiple(int64_t(-6), int64_t(-2)), int64_t(-6));
    CHECK_EQ(round_up_to_multiple(int64_t(103), int64_t(5)), int64_t(105));
    CHECK_EQ(round_up_to_multiple(int64_t(0), int64_t(5)), int64_t(0));
}

TEST_CASE("integer_fun, round_up_to_power_of_two") {
    CHECK_EQ(round_up_to_power_of_two(uint32_t(0)), 1);
    CHECK_EQ(round_up_to_power_of_two(uint32_t(5)), 8);
    CHECK_EQ(round_up_to_power_of_two(uint32_t(1024)), 1024);
    CHECK_EQ(round_up_to_power_of_two(uint32_t(10000)), 16384);
    CHECK_THROWS(round_up_to_power_of_two(std::numeric_limits<uint32_t>::max()));

    CHECK_EQ(round_up_to_power_of_two(uint64_t(0)), 1);
    CHECK_EQ(round_up_to_power_of_two(uint64_t(5)), 8);
    CHECK_EQ(round_up_to_power_of_two(uint64_t(1024)), 1024);
    CHECK_EQ(round_up_to_power_of_two(uint64_t(10000)), 16384);
    CHECK_THROWS(round_up_to_power_of_two(std::numeric_limits<uint64_t>::max()));

    CHECK_EQ(round_up_to_power_of_two(int32_t(0)), 1);
    CHECK_EQ(round_up_to_power_of_two(int32_t(5)), 8);
    CHECK_EQ(round_up_to_power_of_two(int32_t(-5)), 1);
    CHECK_EQ(round_up_to_power_of_two(int32_t(1024)), 1024);
    CHECK_EQ(round_up_to_power_of_two(int32_t(-1024)), 1);
    CHECK_EQ(round_up_to_power_of_two(int32_t(10000)), 16384);
    CHECK_EQ(round_up_to_power_of_two(int32_t(-10000)), 1);
    CHECK_EQ(round_up_to_power_of_two(std::numeric_limits<int32_t>::min()), 1);
    CHECK_THROWS(round_up_to_power_of_two(std::numeric_limits<int32_t>::max()));

    CHECK_EQ(round_up_to_power_of_two(int64_t(0)), 1);
    CHECK_EQ(round_up_to_power_of_two(int64_t(5)), 8);
    CHECK_EQ(round_up_to_power_of_two(int64_t(-5)), 1);
    CHECK_EQ(round_up_to_power_of_two(int64_t(1024)), 1024);
    CHECK_EQ(round_up_to_power_of_two(int64_t(-1024)), 1);
    CHECK_EQ(round_up_to_power_of_two(int64_t(10000)), 16384);
    CHECK_EQ(round_up_to_power_of_two(int64_t(-10000)), 1);
    CHECK_EQ(round_up_to_power_of_two(std::numeric_limits<int64_t>::min()), 1);
    CHECK_THROWS(round_up_to_power_of_two(std::numeric_limits<int64_t>::max()));
}

TEST_CASE("integer_fun, div_ceil and div_floor") {
#define CHECK_DIV_CASE(T, A, B)                                          \
    CHECK_EQ(div_ceil(T(A), T(B)), T(std::ceil(double(A) / double(B)))); \
    CHECK_EQ(div_floor(T(A), T(B)), T(std::floor(double(A) / double(B))));

    CHECK_DIV_CASE(uint32_t, 5, 2);
    CHECK_DIV_CASE(uint32_t, 6, 2);
    CHECK_DIV_CASE(uint32_t, 7, 2);
    CHECK_DIV_CASE(uint32_t, 103, 5);
    CHECK_DIV_CASE(uint32_t, 0, 5);

    CHECK_DIV_CASE(uint64_t, 5, 2);
    CHECK_DIV_CASE(uint64_t, 103, 5);
    CHECK_DIV_CASE(uint64_t, 0, 5);

    CHECK_DIV_CASE(int32_t, 5, 2);
    CHECK_DIV_CASE(int32_t, 5, -2);
    CHECK_DIV_CASE(int32_t, -5, 2);
    CHECK_DIV_CASE(int32_t, -5, -2);
    CHECK_DIV_CASE(int32_t, 6, 2);
    CHECK_DIV_CASE(int32_t, 6, -2);
    CHECK_DIV_CASE(int32_t, -6, 2);
    CHECK_DIV_CASE(int32_t, -6, -2);
    CHECK_DIV_CASE(int32_t, 103, 5);
    CHECK_DIV_CASE(int32_t, 0, 5);

    CHECK_DIV_CASE(int64_t, 5, 2);
    CHECK_DIV_CASE(int64_t, 5, -2);
    CHECK_DIV_CASE(int64_t, -5, 2);
    CHECK_DIV_CASE(int64_t, -5, -2);
    CHECK_DIV_CASE(int64_t, 6, 2);
    CHECK_DIV_CASE(int64_t, 6, -2);
    CHECK_DIV_CASE(int64_t, -6, 2);
    CHECK_DIV_CASE(int64_t, -6, -2);
    CHECK_DIV_CASE(int64_t, 103, 5);
    CHECK_DIV_CASE(int64_t, 0, 5);
}

TEST_CASE("integer_fun, is_power_of_two") {
    CHECK_FALSE(is_power_of_two(uint32_t(0)));
    CHECK(is_power_of_two(uint32_t(1)));
    CHECK(is_power_of_two(uint32_t(1024)));
    CHECK_FALSE(is_power_of_two(uint32_t(1050)));
    CHECK_FALSE(is_power_of_two(uint32_t(std::numeric_limits<uint32_t>::max())));

    CHECK_FALSE(is_power_of_two(int32_t(0)));
    CHECK(is_power_of_two(int32_t(1)));
    CHECK(is_power_of_two(int32_t(1024)));
    CHECK_FALSE(is_power_of_two(int32_t(1050)));
    CHECK_FALSE(is_power_of_two(int32_t(std::numeric_limits<int32_t>::max())));

    CHECK_FALSE(is_power_of_two(uint64_t(0)));
    CHECK(is_power_of_two(uint64_t(1)));
    CHECK(is_power_of_two(uint64_t(1024)));
    CHECK_FALSE(is_power_of_two(uint64_t(1050)));
    CHECK_FALSE(is_power_of_two(uint64_t(std::numeric_limits<uint64_t>::max())));

    CHECK_FALSE(is_power_of_two(int64_t(0)));
    CHECK(is_power_of_two(int64_t(1)));
    CHECK(is_power_of_two(int64_t(1024)));
    CHECK_FALSE(is_power_of_two(int64_t(1050)));
    CHECK_FALSE(is_power_of_two(int64_t(std::numeric_limits<int64_t>::max())));
}

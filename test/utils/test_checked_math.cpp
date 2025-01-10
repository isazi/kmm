#include "catch2/catch_all.hpp"

#include "kmm/utils/checked_math.hpp"
#define CHECK_EQ(A, B) CHECK((A) == (B))

using namespace kmm;

template<typename T>
static constexpr T MAX = std::numeric_limits<T>::max();

template<typename T>
static constexpr T MIN = std::numeric_limits<T>::min();

TEST_CASE("checked math, checked_add") {
    CHECK_EQ(checked_add(size_t(1), size_t(2)), size_t(3));
    CHECK_EQ(checked_add(MAX<size_t>, size_t(0)), MAX<size_t>);
    CHECK_THROWS(checked_add(MAX<size_t>, size_t(1)));

    CHECK_EQ(checked_add(uint32_t(1), uint32_t(2)), uint32_t(3));
    CHECK_EQ(checked_add(MAX<uint32_t>, uint32_t(0)), MAX<uint32_t>);
    CHECK_THROWS(checked_add(MAX<uint32_t>, uint32_t(1)));

    CHECK_EQ(checked_add(int32_t(1), int32_t(2)), int32_t(3));
    CHECK_EQ(checked_add(MAX<int32_t>, int32_t(0)), MAX<int32_t>);
    CHECK_EQ(checked_add(MIN<int32_t>, int32_t(0)), MIN<int32_t>);
    CHECK_THROWS(checked_add(MAX<int32_t>, int32_t(1)));
    CHECK_THROWS(checked_add(MIN<int32_t>, int32_t(-1)));

    CHECK_EQ(checked_add(int8_t(1), int8_t(2)), int8_t(3));
    CHECK_EQ(checked_add(MAX<int8_t>, int8_t(0)), MAX<int8_t>);
    CHECK_EQ(checked_add(MIN<int8_t>, int8_t(0)), MIN<int8_t>);
    CHECK_THROWS(checked_add(MAX<int8_t>, int8_t(1)));
    CHECK_THROWS(checked_add(MIN<int8_t>, int8_t(-1)));
}

TEST_CASE("checked math, checked_mul") {
    CHECK_EQ(checked_mul(size_t(2), size_t(0)), size_t(0));
    CHECK_EQ(checked_mul(size_t(2), size_t(3)), size_t(6));
    CHECK_EQ(checked_mul(size_t(2), MAX<size_t> / 2), MAX<size_t> - 1);
    CHECK_EQ(checked_mul(size_t(1), MAX<size_t>), MAX<size_t>);
    CHECK_THROWS(checked_add(MAX<size_t>, size_t(2)));

    CHECK_EQ(checked_mul(uint32_t(2), uint32_t(0)), uint32_t(0));
    CHECK_EQ(checked_mul(uint32_t(2), uint32_t(3)), uint32_t(6));
    CHECK_EQ(checked_mul(uint32_t(2), MAX<uint32_t> / 2), MAX<uint32_t> - 1);
    CHECK_EQ(checked_mul(uint32_t(1), MAX<uint32_t>), MAX<uint32_t>);
    CHECK_THROWS(checked_mul(MAX<uint32_t>, uint32_t(2)));

    CHECK_EQ(checked_mul(int32_t(2), int32_t(0)), int32_t(0));
    CHECK_EQ(checked_mul(int32_t(2), int32_t(3)), int32_t(6));
    CHECK_EQ(checked_mul(int32_t(2), MAX<int32_t> / 2), MAX<int32_t> - 1);
    CHECK_EQ(checked_mul(int32_t(2), MIN<int32_t> / 2), MIN<int32_t>);
    CHECK_EQ(checked_mul(int32_t(1), MAX<int32_t>), MAX<int32_t>);
    CHECK_EQ(checked_mul(int32_t(1), MIN<int32_t>), MIN<int32_t>);
    CHECK_THROWS(checked_mul(MAX<int32_t>, int32_t(2)));
    CHECK_THROWS(checked_mul(MIN<int32_t>, int32_t(2)));

    CHECK_EQ(checked_mul(int8_t(2), int8_t(0)), int8_t(0));
    CHECK_EQ(checked_mul(int8_t(2), int8_t(3)), int8_t(6));
    CHECK_EQ(checked_mul(int8_t(2), int8_t(MAX<int8_t> / 2)), MAX<int8_t> - 1);
    CHECK_EQ(checked_mul(int8_t(2), int8_t(MIN<int8_t> / 2)), MIN<int8_t>);
    CHECK_EQ(checked_mul(int8_t(1), MAX<int8_t>), MAX<int8_t>);
    CHECK_EQ(checked_mul(int8_t(1), MIN<int8_t>), MIN<int8_t>);
    CHECK_THROWS(checked_mul(MAX<int8_t>, int8_t(2)));
    CHECK_THROWS(checked_mul(MIN<int8_t>, int8_t(2)));
}

TEST_CASE("checked math, checked less") {
    CHECK(compare_less(int32_t(10), int32_t(50)));
    CHECK(compare_less(uint32_t(10), int32_t(50)));
    CHECK(compare_less(size_t(10), int32_t(50)));
    CHECK(compare_less(char(10), uint32_t(50)));
    CHECK(compare_less(int32_t(10), uint32_t(50)));
    CHECK(compare_less(uint32_t(10), uint32_t(50)));
    CHECK(compare_less(size_t(10), uint32_t(50)));
    CHECK(compare_less(char(10), uint32_t(50)));
    CHECK(compare_less(int32_t(10), size_t(50)));
    CHECK(compare_less(uint32_t(10), size_t(50)));
    CHECK(compare_less(size_t(10), size_t(50)));
    CHECK(compare_less(char(10), size_t(50)));
    CHECK(compare_less(int32_t(10), char(50)));
    CHECK(compare_less(uint32_t(10), char(50)));
    CHECK(compare_less(size_t(10), char(50)));
    CHECK(compare_less(char(10), char(50)));

    CHECK_FALSE(compare_less(int32_t(50), int32_t(10)));
    CHECK_FALSE(compare_less(uint32_t(50), int32_t(10)));
    CHECK_FALSE(compare_less(size_t(50), int32_t(10)));
    CHECK_FALSE(compare_less(char(50), uint32_t(10)));
    CHECK_FALSE(compare_less(int32_t(50), uint32_t(10)));
    CHECK_FALSE(compare_less(uint32_t(50), uint32_t(10)));
    CHECK_FALSE(compare_less(size_t(50), uint32_t(10)));
    CHECK_FALSE(compare_less(char(50), uint32_t(10)));
    CHECK_FALSE(compare_less(int32_t(50), size_t(10)));
    CHECK_FALSE(compare_less(uint32_t(50), size_t(10)));
    CHECK_FALSE(compare_less(size_t(50), size_t(10)));
    CHECK_FALSE(compare_less(char(50), size_t(10)));
    CHECK_FALSE(compare_less(int32_t(50), char(10)));
    CHECK_FALSE(compare_less(uint32_t(50), char(10)));
    CHECK_FALSE(compare_less(size_t(50), char(10)));
    CHECK_FALSE(compare_less(char(50), char(10)));

    CHECK(compare_less(int32_t(-1), int32_t(1)));
    //    CHECK(compare_less(uint32_t(-1), int32_t(1)));
    //    CHECK(compare_less(size_t(-1), int32_t(1)));
    CHECK(compare_less(char(-1), uint32_t(1)));
    CHECK(compare_less(int32_t(-1), uint32_t(1)));
    //    CHECK(compare_less(uint32_t(-1), uint32_t(1)));
    //    CHECK(compare_less(size_t(-1), uint32_t(1)));
    CHECK(compare_less(char(-1), uint32_t(1)));
    CHECK(compare_less(int32_t(-1), size_t(1)));
    //    CHECK(compare_less(uint32_t(-1), size_t(1)));
    //    CHECK(compare_less(size_t(-1), size_t(1)));
    //    CHECK(compare_less(char(-1), size_t(1)));
    CHECK(compare_less(int32_t(-1), char(1)));
    //    CHECK(compare_less(uint32_t(-1), char(1)));
    //    CHECK(compare_less(size_t(-1), char(1)));
    CHECK(compare_less(char(-1), char(1)));

    CHECK_FALSE(compare_less(int32_t(1), int32_t(-1)));
    CHECK_FALSE(compare_less(uint32_t(1), int32_t(-1)));
    CHECK_FALSE(compare_less(size_t(1), int32_t(-1)));
    //    CHECK_FALSE(compare_less(char(1), uint32_t(-1)));
    //    CHECK_FALSE(compare_less(int32_t(1), uint32_t(-1)));
    //    CHECK_FALSE(compare_less(uint32_t(1), uint32_t(-1)));
    //    CHECK_FALSE(compare_less(size_t(1), uint32_t(-1)));
    //    CHECK_FALSE(compare_less(char(1), uint32_t(-1)));
    //    CHECK_FALSE(compare_less(int32_t(1), size_t(-1)));
    //    CHECK_FALSE(compare_less(uint32_t(1), size_t(-1)));
    //    CHECK_FALSE(compare_less(size_t(1), size_t(-1)));
    //    CHECK_FALSE(compare_less(char(1), size_t(-1)));
    CHECK_FALSE(compare_less(int32_t(1), char(-1)));
    CHECK_FALSE(compare_less(uint32_t(1), char(-1)));
    CHECK_FALSE(compare_less(size_t(1), char(-1)));
    CHECK_FALSE(compare_less(char(1), char(-1)));

    CHECK(compare_less(MIN<uint32_t>, int32_t(1)));
    CHECK(compare_less(MIN<uint32_t>, int32_t(1)));
    CHECK(compare_less(MIN<size_t>, int32_t(1)));
    CHECK(compare_less(MIN<char>, uint32_t(1)));
    CHECK(compare_less(MIN<int32_t>, uint32_t(1)));
    CHECK(compare_less(MIN<uint32_t>, uint32_t(1)));
    CHECK(compare_less(MIN<size_t>, uint32_t(1)));
    CHECK(compare_less(MIN<char>, uint32_t(1)));
    CHECK(compare_less(MIN<int32_t>, size_t(1)));
    CHECK(compare_less(MIN<uint32_t>, size_t(1)));
    CHECK(compare_less(MIN<size_t>, size_t(1)));
    CHECK(compare_less(MIN<char>, size_t(1)));
    CHECK(compare_less(MIN<int32_t>, char(1)));
    CHECK(compare_less(MIN<uint32_t>, char(1)));
    CHECK(compare_less(MIN<size_t>, char(1)));
    CHECK(compare_less(MIN<char>, char(1)));

    CHECK_FALSE(compare_less(MAX<uint32_t>, int32_t(1)));
    CHECK_FALSE(compare_less(MAX<uint32_t>, int32_t(1)));
    CHECK_FALSE(compare_less(MAX<size_t>, int32_t(1)));
    CHECK_FALSE(compare_less(MAX<char>, uint32_t(1)));
    CHECK_FALSE(compare_less(MAX<int32_t>, uint32_t(1)));
    CHECK_FALSE(compare_less(MAX<uint32_t>, uint32_t(1)));
    CHECK_FALSE(compare_less(MAX<size_t>, uint32_t(1)));
    CHECK_FALSE(compare_less(MAX<char>, uint32_t(1)));
    CHECK_FALSE(compare_less(MAX<int32_t>, size_t(1)));
    CHECK_FALSE(compare_less(MAX<uint32_t>, size_t(1)));
    CHECK_FALSE(compare_less(MAX<size_t>, size_t(1)));
    CHECK_FALSE(compare_less(MAX<char>, size_t(1)));
    CHECK_FALSE(compare_less(MAX<int32_t>, char(1)));
    CHECK_FALSE(compare_less(MAX<uint32_t>, char(1)));
    CHECK_FALSE(compare_less(MAX<size_t>, char(1)));
    CHECK_FALSE(compare_less(MAX<char>, char(1)));
}

TEST_CASE("checked math, checked cast") {
    CHECK_EQ(checked_cast<int32_t>(int32_t(1)), int32_t(1));
    CHECK_EQ(checked_cast<int32_t>(int32_t(-1)), int32_t(-1));
    CHECK_EQ(checked_cast<int32_t>(MIN<int32_t>), MIN<int32_t>);
    CHECK_EQ(checked_cast<int32_t>(MAX<int32_t>), MAX<int32_t>);

    CHECK_EQ(checked_cast<int32_t>(uint32_t(1)), uint32_t(1));
    CHECK_THROWS(checked_cast<int32_t>(uint32_t(-1)));
    CHECK_EQ(checked_cast<int32_t>(MIN<uint32_t>), MIN<uint32_t>);
    CHECK_THROWS(checked_cast<int32_t>(MAX<uint32_t>));

    CHECK_EQ(checked_cast<int32_t>(size_t(1)), size_t(1));
    CHECK_THROWS(checked_cast<int32_t>(size_t(-1)));
    CHECK_EQ(checked_cast<int32_t>(MIN<size_t>), MIN<size_t>);
    CHECK_THROWS(checked_cast<int32_t>(MAX<size_t>));

    CHECK_EQ(checked_cast<int32_t>(char(1)), int32_t(1));
    CHECK_EQ(checked_cast<int32_t>(char(-1)), int32_t(-1));
    CHECK_EQ(checked_cast<int32_t>(MIN<char>), int32_t(MIN<char>));
    CHECK_EQ(checked_cast<int32_t>(MAX<char>), int32_t(MAX<char>));

    CHECK_EQ(checked_cast<uint32_t>(int32_t(1)), uint32_t(1));
    CHECK_THROWS(checked_cast<uint32_t>(int32_t(-1)));
    CHECK_THROWS(checked_cast<uint32_t>(MIN<int32_t>));
    CHECK_EQ(checked_cast<uint32_t>(MAX<int32_t>), uint32_t(MAX<int32_t>));

    CHECK_EQ(checked_cast<uint32_t>(uint32_t(1)), uint32_t(1));
    CHECK_EQ(checked_cast<uint32_t>(MIN<uint32_t>), MIN<uint32_t>);
    CHECK_EQ(checked_cast<uint32_t>(MAX<uint32_t>), MAX<uint32_t>);

    CHECK_EQ(checked_cast<uint32_t>(size_t(1)), uint32_t(1));
    CHECK_THROWS(checked_cast<uint32_t>(size_t(-1)));
    CHECK_EQ(checked_cast<uint32_t>(MIN<size_t>), uint32_t(MIN<size_t>));
    CHECK_THROWS(checked_cast<uint32_t>(MAX<size_t>));

    CHECK_EQ(checked_cast<uint32_t>(char(1)), uint32_t(1));
    CHECK_THROWS(checked_cast<uint32_t>(char(-1)));
    CHECK_THROWS(checked_cast<uint32_t>(MIN<char>));
    CHECK_EQ(checked_cast<uint32_t>(MAX<char>), uint32_t(MAX<char>));

    CHECK_EQ(checked_cast<char>(int32_t(1)), char(1));
    CHECK_EQ(checked_cast<char>(int32_t(-1)), char(-1));
    CHECK_THROWS(checked_cast<char>(MIN<int32_t>));
    CHECK_THROWS(checked_cast<char>(MAX<int32_t>));

    CHECK_EQ(checked_cast<char>(uint32_t(1)), char(1));
    CHECK_THROWS(checked_cast<char>(uint32_t(-1)));
    CHECK_EQ(checked_cast<char>(MIN<uint32_t>), char(MIN<uint32_t>));
    CHECK_THROWS(checked_cast<char>(MAX<uint32_t>));

    CHECK_EQ(checked_cast<char>(size_t(1)), char(1));
    CHECK_THROWS(checked_cast<char>(size_t(-1)));
    CHECK_EQ(checked_cast<char>(MIN<size_t>), char(MIN<size_t>));
    CHECK_THROWS(checked_cast<char>(MAX<size_t>));

    CHECK_EQ(checked_cast<char>(char(1)), char(1));
    CHECK_EQ(checked_cast<char>(char(-1)), char(-1));
    CHECK_EQ(checked_cast<char>(MIN<char>), MIN<char>);
    CHECK_EQ(checked_cast<char>(MAX<char>), MAX<char>);

    CHECK_EQ(checked_cast<size_t>(int32_t(1)), size_t(1));
    CHECK_THROWS(checked_cast<size_t>(int32_t(-1)));
    CHECK_THROWS(checked_cast<size_t>(MIN<int32_t>));
    CHECK_EQ(checked_cast<size_t>(MAX<int32_t>), size_t(MAX<int32_t>));

    CHECK_EQ(checked_cast<size_t>(uint32_t(1)), size_t(1));
    CHECK_EQ(checked_cast<size_t>(MIN<uint32_t>), size_t(MIN<uint32_t>));
    CHECK_EQ(checked_cast<size_t>(MAX<uint32_t>), size_t(MAX<uint32_t>));

    CHECK_EQ(checked_cast<size_t>(size_t(1)), size_t(1));
    CHECK_EQ(checked_cast<size_t>(size_t(-1)), size_t(-1));
    CHECK_EQ(checked_cast<size_t>(MIN<size_t>), MIN<size_t>);
    CHECK_EQ(checked_cast<size_t>(MAX<size_t>), MAX<size_t>);

    CHECK_EQ(checked_cast<size_t>(char(1)), size_t(1));
    CHECK_THROWS(checked_cast<size_t>(char(-1)));
    CHECK_THROWS(checked_cast<size_t>(MIN<char>));
    CHECK_EQ(checked_cast<size_t>(MAX<char>), size_t(MAX<char>));
}
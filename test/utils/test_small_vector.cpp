#include <cmath>

#include "catch2/catch_all.hpp"

#include "kmm/utils/small_vector.hpp"
#define CHECK_EQ(A, B) CHECK((A) == (B))

using namespace kmm;

TEST_CASE("small_vector, basics") {
    small_vector<int, 2> x;
    CHECK_EQ(x.size(), 0);
    CHECK_EQ(x.capacity(), 2);
    CHECK_EQ(x.is_empty(), true);
    CHECK_EQ(x.is_heap_allocated(), false);

    x.push_back(42);
    CHECK_EQ(x.size(), 1);
    CHECK_EQ(x.capacity(), 2);
    CHECK_EQ(x.is_empty(), false);
    CHECK_EQ(x.is_heap_allocated(), false);
    CHECK_EQ(x[0], 42);

    x.push_back(57);
    CHECK_EQ(x.size(), 2);
    CHECK_EQ(x.capacity(), 2);
    CHECK_EQ(x.is_empty(), false);
    CHECK_EQ(x.is_heap_allocated(), false);
    CHECK_EQ(x[0], 42);
    CHECK_EQ(x[1], 57);

    x.push_back(64);
    CHECK_EQ(x.size(), 3);
    CHECK_EQ(x.capacity(), 16);
    CHECK_EQ(x.is_empty(), false);
    CHECK_EQ(x.is_heap_allocated(), true);
    CHECK_EQ(x[0], 42);
    CHECK_EQ(x[1], 57);
    CHECK_EQ(x[2], 64);

    x.clear();
    CHECK_EQ(x.size(), 0);
    CHECK_EQ(x.capacity(), 16);
    CHECK_EQ(x.is_empty(), true);
    CHECK_EQ(x.is_heap_allocated(), true);
}

TEST_CASE("small_vector, list_initializer") {
    SECTION("empty list") {
        small_vector<int, 4> x = {};
        CHECK_EQ(x.size(), 0);
        CHECK_EQ(x.capacity(), 4);
        CHECK_EQ(x.is_heap_allocated(), false);
    }

    SECTION("stack allocated") {
        small_vector<int, 4> y = {1, 2, 3};
        CHECK_EQ(y.size(), 3);
        CHECK_EQ(y.capacity(), 4);
        CHECK_EQ(y.is_heap_allocated(), false);
        CHECK_EQ(y[0], 1);
        CHECK_EQ(y[1], 2);
        CHECK_EQ(y[2], 3);
    }

    SECTION("heap allocated") {
        small_vector<int, 4> z = {1, 2, 3, 4, 5};
        CHECK(z.size() == 5);
        CHECK(z.capacity() == 16);
        CHECK(z.is_heap_allocated());
        CHECK(z[0] == 1);
        CHECK(z[1] == 2);
        CHECK(z[2] == 3);
        CHECK(z[3] == 4);
        CHECK(z[4] == 5);
    }
}

TEST_CASE("small_vector, copy constructor") {
    {
        small_vector<int, 2> x = {1, 2, 3};
        small_vector<int, 2> y = x;

        CHECK_EQ(x.size(), 3);
        CHECK_EQ(y.size(), 3);
        CHECK_EQ(x.is_heap_allocated(), true);
        CHECK_EQ(y.is_heap_allocated(), true);

        CHECK_EQ(x[0], 1);
        CHECK_EQ(x[1], 2);
        CHECK_EQ(x[2], 3);

        CHECK_EQ(y[0], 1);
        CHECK_EQ(y[1], 2);
        CHECK_EQ(y[2], 3);
    }

    {
        small_vector<int, 4> x = {1, 2, 3};
        small_vector<int, 4> y = x;

        CHECK_EQ(x.size(), 3);
        CHECK_EQ(y.size(), 3);
        CHECK_EQ(x.is_heap_allocated(), false);
        CHECK_EQ(y.is_heap_allocated(), false);

        CHECK_EQ(x[0], 1);
        CHECK_EQ(x[1], 2);
        CHECK_EQ(x[2], 3);

        CHECK_EQ(y[0], 1);
        CHECK_EQ(y[1], 2);
        CHECK_EQ(y[2], 3);
    }

    {
        small_vector<int, 2> x = {1, 2, 3};
        small_vector<int, 4> y = x;

        CHECK_EQ(x.size(), 3);
        CHECK_EQ(y.size(), 3);
        CHECK_EQ(x.is_heap_allocated(), true);
        CHECK_EQ(y.is_heap_allocated(), false);

        CHECK_EQ(x[0], 1);
        CHECK_EQ(x[1], 2);
        CHECK_EQ(x[2], 3);

        CHECK_EQ(y[0], 1);
        CHECK_EQ(y[1], 2);
        CHECK_EQ(y[2], 3);
    }

    {
        small_vector<int, 4> x = {1, 2, 3};
        small_vector<int, 2> y = x;

        CHECK_EQ(x.size(), 3);
        CHECK_EQ(y.size(), 3);
        CHECK_EQ(x.is_heap_allocated(), false);
        CHECK_EQ(y.is_heap_allocated(), true);

        CHECK_EQ(x[0], 1);
        CHECK_EQ(x[1], 2);
        CHECK_EQ(x[2], 3);

        CHECK_EQ(y[0], 1);
        CHECK_EQ(y[1], 2);
        CHECK_EQ(y[2], 3);
    }
}

TEST_CASE("small_vector, move constructor") {
    SECTION("heap allocated") {
        small_vector<int, 2> x = {1, 2, 3};
        small_vector<int, 2> y = std::move(x);

        CHECK_EQ(x.size(), 0);
        CHECK_EQ(y.size(), 3);
        CHECK_EQ(x.is_heap_allocated(), false);
        CHECK_EQ(y.is_heap_allocated(), true);

        CHECK_EQ(y[0], 1);
        CHECK_EQ(y[1], 2);
        CHECK_EQ(y[2], 3);
    }

    SECTION("stack allocated") {
        small_vector<int, 4> x = {1, 2, 3};
        small_vector<int, 4> y = std::move(x);

        CHECK_EQ(x.size(), 0);
        CHECK_EQ(y.size(), 3);
        CHECK_EQ(x.is_heap_allocated(), false);
        CHECK_EQ(y.is_heap_allocated(), false);

        CHECK_EQ(y[0], 1);
        CHECK_EQ(y[1], 2);
        CHECK_EQ(y[2], 3);
    }
}

TEST_CASE("small_vector, contains") {
    small_vector<int, 4> x = {1, 2, 3};
    CHECK_EQ(x.contains(2), true);
    CHECK_EQ(x.contains(42), false);

    CHECK_EQ(x.find_if([&](auto v) { return v == 2; }), &x[1]);
    CHECK_EQ(x.find_if([&](auto v) { return v == 42; }), x.end());
}

TEST_CASE("small_vector, iterate") {
    small_vector<int, 4> x = {0, 1, 2, 3, 4, 5, 6, 7, 8};

    int index = 0;
    for (const auto& v : x) {
        CHECK_EQ(v, index);
        index++;
    }

    CHECK_EQ(index, 9);
}
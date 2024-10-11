#include <gtest/gtest.h>
#include <cmath>

#include "kmm/utils/small_vector.hpp"

using namespace kmm;

TEST(SmallVector, basics) {
    small_vector<int, 2> x;
    ASSERT_EQ(x.size(), 0);
    ASSERT_EQ(x.capacity(), 2);
    ASSERT_EQ(x.is_empty(), true);
    ASSERT_EQ(x.is_heap_allocated(), false);

    x.push_back(42);
    ASSERT_EQ(x.size(), 1);
    ASSERT_EQ(x.capacity(), 2);
    ASSERT_EQ(x.is_empty(), false);
    ASSERT_EQ(x.is_heap_allocated(), false);
    ASSERT_EQ(x[0], 42);

    x.push_back(57);
    ASSERT_EQ(x.size(), 2);
    ASSERT_EQ(x.capacity(), 2);
    ASSERT_EQ(x.is_empty(), false);
    ASSERT_EQ(x.is_heap_allocated(), false);
    ASSERT_EQ(x[0], 42);
    ASSERT_EQ(x[1], 57);

    x.push_back(64);
    ASSERT_EQ(x.size(), 3);
    ASSERT_EQ(x.capacity(), 16);
    ASSERT_EQ(x.is_empty(), false);
    ASSERT_EQ(x.is_heap_allocated(), true);
    ASSERT_EQ(x[0], 42);
    ASSERT_EQ(x[1], 57);
    ASSERT_EQ(x[2], 64);

    x.clear();
    ASSERT_EQ(x.size(), 0);
    ASSERT_EQ(x.capacity(), 16);
    ASSERT_EQ(x.is_empty(), true);
    ASSERT_EQ(x.is_heap_allocated(), true);
}

TEST(SmallVector, list_initializer) {
    small_vector<int, 4> x = {};
    ASSERT_EQ(x.size(), 0);
    ASSERT_EQ(x.capacity(), 4);
    ASSERT_EQ(x.is_heap_allocated(), false);

    small_vector<int, 4> y = {1, 2, 3};
    ASSERT_EQ(y.size(), 3);
    ASSERT_EQ(y.capacity(), 4);
    ASSERT_EQ(x.is_heap_allocated(), false);
    ASSERT_EQ(y[0], 1);
    ASSERT_EQ(y[1], 2);
    ASSERT_EQ(y[2], 3);

    small_vector<int, 4> z = {1, 2, 3, 4, 5};
    ASSERT_EQ(z.size(), 5);
    ASSERT_EQ(z.capacity(), 16);
    ASSERT_EQ(z.is_heap_allocated(), true);
    ASSERT_EQ(z[0], 1);
    ASSERT_EQ(z[1], 2);
    ASSERT_EQ(z[2], 3);
    ASSERT_EQ(z[3], 4);
    ASSERT_EQ(z[4], 5);
}

TEST(SmallVector, copy_constructor) {
    {
        small_vector<int, 2> x = {1, 2, 3};
        small_vector<int, 2> y = x;

        ASSERT_EQ(x.size(), 3);
        ASSERT_EQ(y.size(), 3);
        ASSERT_EQ(x.is_heap_allocated(), true);
        ASSERT_EQ(y.is_heap_allocated(), true);

        ASSERT_EQ(x[0], 1);
        ASSERT_EQ(x[1], 2);
        ASSERT_EQ(x[2], 3);

        ASSERT_EQ(y[0], 1);
        ASSERT_EQ(y[1], 2);
        ASSERT_EQ(y[2], 3);
    }

    {
        small_vector<int, 4> x = {1, 2, 3};
        small_vector<int, 4> y = x;

        ASSERT_EQ(x.size(), 3);
        ASSERT_EQ(y.size(), 3);
        ASSERT_EQ(x.is_heap_allocated(), false);
        ASSERT_EQ(y.is_heap_allocated(), false);

        ASSERT_EQ(x[0], 1);
        ASSERT_EQ(x[1], 2);
        ASSERT_EQ(x[2], 3);

        ASSERT_EQ(y[0], 1);
        ASSERT_EQ(y[1], 2);
        ASSERT_EQ(y[2], 3);
    }

    {
        small_vector<int, 2> x = {1, 2, 3};
        small_vector<int, 4> y = x;

        ASSERT_EQ(x.size(), 3);
        ASSERT_EQ(y.size(), 3);
        ASSERT_EQ(x.is_heap_allocated(), true);
        ASSERT_EQ(y.is_heap_allocated(), false);

        ASSERT_EQ(x[0], 1);
        ASSERT_EQ(x[1], 2);
        ASSERT_EQ(x[2], 3);

        ASSERT_EQ(y[0], 1);
        ASSERT_EQ(y[1], 2);
        ASSERT_EQ(y[2], 3);
    }

    {
        small_vector<int, 4> x = {1, 2, 3};
        small_vector<int, 2> y = x;

        ASSERT_EQ(x.size(), 3);
        ASSERT_EQ(y.size(), 3);
        ASSERT_EQ(x.is_heap_allocated(), false);
        ASSERT_EQ(y.is_heap_allocated(), true);

        ASSERT_EQ(x[0], 1);
        ASSERT_EQ(x[1], 2);
        ASSERT_EQ(x[2], 3);

        ASSERT_EQ(y[0], 1);
        ASSERT_EQ(y[1], 2);
        ASSERT_EQ(y[2], 3);
    }
}

TEST(SmallVector, move_constructor) {
    {
        small_vector<int, 2> x = {1, 2, 3};
        small_vector<int, 2> y = std::move(x);

        ASSERT_EQ(x.size(), 0);
        ASSERT_EQ(y.size(), 3);
        ASSERT_EQ(x.is_heap_allocated(), false);
        ASSERT_EQ(y.is_heap_allocated(), true);

        ASSERT_EQ(y[0], 1);
        ASSERT_EQ(y[1], 2);
        ASSERT_EQ(y[2], 3);
    }

    {
        small_vector<int, 4> x = {1, 2, 3};
        small_vector<int, 4> y = std::move(x);

        ASSERT_EQ(x.size(), 0);
        ASSERT_EQ(y.size(), 3);
        ASSERT_EQ(x.is_heap_allocated(), false);
        ASSERT_EQ(y.is_heap_allocated(), false);

        ASSERT_EQ(y[0], 1);
        ASSERT_EQ(y[1], 2);
        ASSERT_EQ(y[2], 3);
    }
}

TEST(SmallVector, contains) {
    small_vector<int, 4> x = {1, 2, 3};
    ASSERT_EQ(x.contains(2), true);
    ASSERT_EQ(x.contains(42), false);

    ASSERT_EQ(x.find_if([&](auto v) { return v == 2; }), &x[1]);
    ASSERT_EQ(x.find_if([&](auto v) { return v == 42; }), x.end());
}

TEST(SmallVector, iterate) {
    small_vector<int> x = {0, 1, 2, 3, 4, 5, 6, 7, 8};

    size_t index = 0;
    for (const auto& v: x) {
        ASSERT_EQ(v, index);
        index++;
    }

    ASSERT_EQ(index, 9);
}
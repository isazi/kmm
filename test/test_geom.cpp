#include <gtest/gtest.h>

#include "kmm/core/geometry.hpp"

using namespace kmm;

TEST(Geometry, point_basics) {
    point<3> x = {};
    ASSERT_EQ(x, point(0, 0, 0));

    point<3> y = {1, 2, 3};
    ASSERT_EQ(y, point(1, 2, 3));

    point<3> z = point<3>::fill(42);
    ASSERT_EQ(z, point(42, 42, 42));

    auto a = point<3, float>::from(point<3, int>{4, 5, 6});
    ASSERT_EQ(a, (point<3, float>(4.0f, 5.0f, 6.0f)));

    point<3> zero = point<3>::zero();
    ASSERT_EQ(zero, point(0, 0, 0));

    point<3> one = point<3>::one();
    ASSERT_EQ(one, point(1, 1, 1));

    one[0] = 42;
    ASSERT_EQ(one, point(42, 1, 1));

    ASSERT_FALSE(one.equals(zero));
    ASSERT_FALSE(zero.equals(one));
    ASSERT_TRUE(zero.equals(zero));
    ASSERT_TRUE(one.equals(one));
}

TEST(Geometry, dim_basics) {
    dim<3> x = {};
    ASSERT_EQ(x, dim(1, 1, 1));
    ASSERT_EQ(x.is_empty(), false);
    ASSERT_EQ(x.volume(), 1);

    dim<3> y = {1, 2, 3};
    ASSERT_EQ(y, dim(1, 2, 3));
    ASSERT_EQ(y.is_empty(), false);
    ASSERT_EQ(y.volume(), 6);

    dim<3> z = {-5, 3, 1};
    ASSERT_EQ(z, dim(-5, 3, 1));
    ASSERT_EQ(z.is_empty(), true);
    ASSERT_EQ(z.volume(), 0);

    auto a = dim<3, float>::from(dim<3, int>{4, 5, 6});
    ASSERT_EQ(a, (dim<3, float>(4.0f, 5.0f, 6.0f)));
    ASSERT_EQ(a.is_empty(), false);
    ASSERT_EQ(a.volume(), 120.0f);

    dim<3> zero = dim<3>::zero();
    ASSERT_EQ(zero, dim(0, 0, 0));
    ASSERT_EQ(zero.is_empty(), true);
    ASSERT_EQ(zero.volume(), 0);

    dim<3> one = dim<3>::one();
    ASSERT_EQ(one, dim(1, 1, 1));
    ASSERT_EQ(one.is_empty(), false);
    ASSERT_EQ(one.volume(), 1);

    one[0] = 42;
    ASSERT_EQ(one, dim(42, 1, 1));
    ASSERT_EQ(one.is_empty(), false);
    ASSERT_EQ(one.volume(), 42);
}

TEST(Geometry, dim_intersection) {
    dim<3> x = {5, 42, 1};
    dim<3> y = {1, 2, 3};
    dim<3> z = {1, 1, 1};
    dim<3> w = {-5, 2, 1};
    
    ASSERT_EQ(x.intersection(x), dim(5, 42, 1));
    ASSERT_EQ(x.intersection(y), dim(1, 2, 1));
    ASSERT_EQ(x.intersection(z), dim(1, 1, 1));
    ASSERT_EQ(x.intersection(w), dim(0, 2, 1));
    
    ASSERT_EQ(y.intersection(x), dim(1, 2, 1));
    ASSERT_EQ(y.intersection(y), dim(1, 2, 3));
    ASSERT_EQ(y.intersection(z), dim(1, 1, 1));
    ASSERT_EQ(y.intersection(w), dim(0, 2, 1));
    
    ASSERT_EQ(z.intersection(x), dim(1, 1, 1));
    ASSERT_EQ(z.intersection(y), dim(1, 1, 1));
    ASSERT_EQ(z.intersection(z), dim(1, 1, 1));
    ASSERT_EQ(z.intersection(w), dim(0, 1, 1));
    
    ASSERT_EQ(w.intersection(x), dim(0, 2, 1));
    ASSERT_EQ(w.intersection(y), dim(0, 2, 1));
    ASSERT_EQ(w.intersection(z), dim(0, 1, 1));
    ASSERT_EQ(w.intersection(w), dim(0, 2, 1));
    
    
    ASSERT_EQ(x.overlaps(x), true);
    ASSERT_EQ(x.overlaps(y), true);
    ASSERT_EQ(x.overlaps(z), true);
    ASSERT_EQ(x.overlaps(w), false);
    
    ASSERT_EQ(y.overlaps(x), true);
    ASSERT_EQ(y.overlaps(y), true);
    ASSERT_EQ(y.overlaps(z), true);
    ASSERT_EQ(y.overlaps(w), false);
    
    ASSERT_EQ(z.overlaps(x), true);
    ASSERT_EQ(z.overlaps(y), true);
    ASSERT_EQ(z.overlaps(z), true);
    ASSERT_EQ(z.overlaps(w), false);
    
    ASSERT_EQ(w.overlaps(x), false);
    ASSERT_EQ(w.overlaps(y), false);
    ASSERT_EQ(w.overlaps(z), false);
    ASSERT_EQ(w.overlaps(w), false);
    
    
    ASSERT_EQ(x.contains(x), true);
    ASSERT_EQ(x.contains(y), false);
    ASSERT_EQ(x.contains(z), true);
    ASSERT_EQ(x.contains(w), true);
    
    ASSERT_EQ(y.contains(x), false);
    ASSERT_EQ(y.contains(y), true);
    ASSERT_EQ(y.contains(z), true);
    ASSERT_EQ(y.contains(w), true);
    
    ASSERT_EQ(z.contains(x), false);
    ASSERT_EQ(z.contains(y), false);
    ASSERT_EQ(z.contains(z), true);
    ASSERT_EQ(z.contains(w), true);

    ASSERT_EQ(w.contains(x), false);
    ASSERT_EQ(w.contains(y), false);
    ASSERT_EQ(w.contains(z), false);
    ASSERT_EQ(w.contains(w), true);


    point<3> a = {0, 0, 0};
    point<3> b = {0, 2, 0};
    point<3> c = {-2, 0, 1};

    ASSERT_EQ(x.contains(a), true);
    ASSERT_EQ(x.contains(b), true);
    ASSERT_EQ(x.contains(c), false);

    ASSERT_EQ(y.contains(a), true);
    ASSERT_EQ(y.contains(b), false);
    ASSERT_EQ(y.contains(c), false);

    ASSERT_EQ(z.contains(a), true);
    ASSERT_EQ(z.contains(b), false);
    ASSERT_EQ(z.contains(c), false);

    ASSERT_EQ(w.contains(a), false);
    ASSERT_EQ(w.contains(b), false);
    ASSERT_EQ(w.contains(c), false);
}

TEST(Geometry, rect_basics) {
    rect<3> a = {{0, 0, 0}, {42, 2, 1}};
    ASSERT_EQ(a.offset, point(0, 0, 0));
    ASSERT_EQ(a.sizes, dim(42, 2, 1));
    ASSERT_EQ(a.begin(), point(0, 0, 0));
    ASSERT_EQ(a.end(), point(42, 2, 1));
    ASSERT_EQ(a.size(), 84);
    ASSERT_EQ(a.is_empty(), false);

    rect<3> b = {{1, 1, 1}, {3, 2, 1}};
    ASSERT_EQ(b.offset, point(1, 1, 1));
    ASSERT_EQ(b.sizes, dim(3, 2, 1));
    ASSERT_EQ(b.begin(), point(1, 1, 1));
    ASSERT_EQ(b.end(), point(4, 3, 2));
    ASSERT_EQ(b.size(), 6);
    ASSERT_EQ(b.is_empty(), false);

    rect<3> c = {{1, -5, 1}, {2, 2, 2}};
    ASSERT_EQ(c.offset, point(1, -5, 1));
    ASSERT_EQ(c.sizes, dim(2, 2, 2));
    ASSERT_EQ(c.begin(), point(1, -5, 1));
    ASSERT_EQ(c.end(), point(3, -3, 3));
    ASSERT_EQ(c.size(), 8);
    ASSERT_EQ(c.is_empty(), false);

    rect<3> d = {{5, 1, 2}, {-5, 3, 1}};
    ASSERT_EQ(d.offset, point(5, 1, 2));
    ASSERT_EQ(d.sizes, dim(-5, 3, 1));
    ASSERT_EQ(d.begin(), point(5, 1, 2));
    ASSERT_EQ(d.end(), point(5, 4, 3));
    ASSERT_EQ(d.size(), 0);
    ASSERT_EQ(d.is_empty(), true);
}

TEST(Geometry, rect_intersection) {
    rect<3> a = {{0, 0, 0}, {42, 2, 5}};
    rect<3> b = {{1, 1, 1}, {3, 1, 1}};
    rect<3> c = {{1, -5, 1}, {2, 20, 2}};
    rect<3> d = {{5, 1, 2}, {-3, 3, 1}};

    ASSERT_EQ(a.intersection(a), a);
    ASSERT_EQ(a.intersection(b), (rect<3>{{1, 1, 1}, {3, 1, 1}}));
    ASSERT_EQ(a.intersection(c), (rect<3>{{1, 0, 1}, {2, 2, 2}}));
    ASSERT_EQ(a.intersection(d),  (rect<3>{{5, 1, 2}, {-3, 1, 1}}));

    ASSERT_EQ(b.intersection(a), (rect<3>{{1, 1, 1}, {3, 1, 1}}));
    ASSERT_EQ(b.intersection(b), b);
    ASSERT_EQ(b.intersection(c), (rect<3>{{1, 1, 1}, {2, 1, 1}}));
    ASSERT_EQ(b.intersection(d), rect<3>());

    ASSERT_EQ(c.intersection(a), (rect<3>{{1, 0, 1}, {2, 2, 2}}));
    ASSERT_EQ(c.intersection(b), (rect<3>{{1, 1, 1}, {2, 1, 1}}));
    ASSERT_EQ(c.intersection(c), c);
    ASSERT_EQ(c.intersection(d), rect<3>());

    ASSERT_EQ(d.intersection(a), (rect<3>{{5, 1, 2}, {-3, 1, 1}}));
    ASSERT_EQ(d.intersection(b), rect<3>());
    ASSERT_EQ(d.intersection(c), rect<3>());
    ASSERT_EQ(d.intersection(d), rect<3>());
    
    
    ASSERT_EQ(a.overlaps(a), true);
    ASSERT_EQ(a.overlaps(b), true);
    ASSERT_EQ(a.overlaps(c), true);
    ASSERT_EQ(a.overlaps(d), false);
    
    ASSERT_EQ(b.overlaps(a), true);
    ASSERT_EQ(b.overlaps(b), true);
    ASSERT_EQ(b.overlaps(c), true);
    ASSERT_EQ(b.overlaps(d), false);
    
    ASSERT_EQ(c.overlaps(a), true);
    ASSERT_EQ(c.overlaps(b), true);
    ASSERT_EQ(c.overlaps(c), true);
    ASSERT_EQ(c.overlaps(d), false);
    
    ASSERT_EQ(d.overlaps(a), false);
    ASSERT_EQ(d.overlaps(b), false);
    ASSERT_EQ(d.overlaps(c), false);
    ASSERT_EQ(d.overlaps(d), false);
    
    
    ASSERT_EQ(a.contains(a), true);
    ASSERT_EQ(a.contains(b), true);
    ASSERT_EQ(a.contains(c), false);
    ASSERT_EQ(a.contains(d), true);
    
    ASSERT_EQ(b.contains(a), false);
    ASSERT_EQ(b.contains(b), true);
    ASSERT_EQ(b.contains(c), false);
    ASSERT_EQ(b.contains(d), true);
    
    ASSERT_EQ(c.contains(a), false);
    ASSERT_EQ(c.contains(b), false);
    ASSERT_EQ(c.contains(c), true);
    ASSERT_EQ(c.contains(d), true);
    
    ASSERT_EQ(d.contains(a), false);
    ASSERT_EQ(d.contains(b), false);
    ASSERT_EQ(d.contains(c), false);
    ASSERT_EQ(d.contains(d), true);
}
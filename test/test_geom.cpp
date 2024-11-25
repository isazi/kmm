#include <gtest/gtest.h>

#include "kmm/core/geometry.hpp"

using namespace kmm;

TEST(Geometry, Point_basics) {
    Point<3> x = {};
    ASSERT_EQ(x, Point(0, 0, 0));

    Point<3> y = {1, 2, 3};
    ASSERT_EQ(y, Point(1, 2, 3));

    Point<3> z = Point<3>::fill(42);
    ASSERT_EQ(z, Point(42, 42, 42));

    auto a = Point<3, float>::from(Point<3, int>{4, 5, 6});
    ASSERT_EQ(a, (Point<3, float>(4.0f, 5.0f, 6.0f)));

    Point<3> zero = Point<3>::zero();
    ASSERT_EQ(zero, Point(0, 0, 0));

    Point<3> one = Point<3>::one();
    ASSERT_EQ(one, Point(1, 1, 1));

    one[0] = 42;
    ASSERT_EQ(one, Point(42, 1, 1));

    ASSERT_FALSE(one == zero);
    ASSERT_FALSE(zero == one);
    ASSERT_TRUE(zero == zero);
    ASSERT_TRUE(one == one);
}

TEST(Geometry, dim_basics) {
    Dim<3> x = {};
    ASSERT_EQ(x, Dim(1, 1, 1));
    ASSERT_EQ(x.is_empty(), false);
    ASSERT_EQ(x.volume(), 1);

    Dim<3> y = {1, 2, 3};
    ASSERT_EQ(y, Dim(1, 2, 3));
    ASSERT_EQ(y.is_empty(), false);
    ASSERT_EQ(y.volume(), 6);

    Dim<3> z = {-5, 3, 1};
    ASSERT_EQ(z, Dim(-5, 3, 1));
    ASSERT_EQ(z.is_empty(), true);
    ASSERT_EQ(z.volume(), 0);

    auto a = Dim<3, float>::from(Dim<3, int>{4, 5, 6});
    ASSERT_EQ(a, (Dim<3, float>(4.0f, 5.0f, 6.0f)));
    ASSERT_EQ(a.is_empty(), false);
    ASSERT_EQ(a.volume(), 120.0f);

    Dim<3> zero = Dim<3>::zero();
    ASSERT_EQ(zero, Dim(0, 0, 0));
    ASSERT_EQ(zero.is_empty(), true);
    ASSERT_EQ(zero.volume(), 0);

    Dim<3> one = Dim<3>::one();
    ASSERT_EQ(one, Dim(1, 1, 1));
    ASSERT_EQ(one.is_empty(), false);
    ASSERT_EQ(one.volume(), 1);

    one[0] = 42;
    ASSERT_EQ(one, Dim(42, 1, 1));
    ASSERT_EQ(one.is_empty(), false);
    ASSERT_EQ(one.volume(), 42);
}

TEST(Geometry, dim_intersection) {
    Dim<3> x = {5, 42, 1};
    Dim<3> y = {1, 2, 3};
    Dim<3> z = {1, 1, 1};
    Dim<3> w = {-5, 2, 1};
    
    ASSERT_EQ(x.intersection(x), Dim(5, 42, 1));
    ASSERT_EQ(x.intersection(y), Dim(1, 2, 1));
    ASSERT_EQ(x.intersection(z), Dim(1, 1, 1));
    ASSERT_EQ(x.intersection(w), Dim(0, 2, 1));
    
    ASSERT_EQ(y.intersection(x), Dim(1, 2, 1));
    ASSERT_EQ(y.intersection(y), Dim(1, 2, 3));
    ASSERT_EQ(y.intersection(z), Dim(1, 1, 1));
    ASSERT_EQ(y.intersection(w), Dim(0, 2, 1));
    
    ASSERT_EQ(z.intersection(x), Dim(1, 1, 1));
    ASSERT_EQ(z.intersection(y), Dim(1, 1, 1));
    ASSERT_EQ(z.intersection(z), Dim(1, 1, 1));
    ASSERT_EQ(z.intersection(w), Dim(0, 1, 1));
    
    ASSERT_EQ(w.intersection(x), Dim(0, 2, 1));
    ASSERT_EQ(w.intersection(y), Dim(0, 2, 1));
    ASSERT_EQ(w.intersection(z), Dim(0, 1, 1));
    ASSERT_EQ(w.intersection(w), Dim(0, 2, 1));
    
    
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


    Point<3> a = {0, 0, 0};
    Point<3> b = {0, 2, 0};
    Point<3> c = {-2, 0, 1};

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

TEST(Geometry, range_basics) {
    Range<3> a = {{0, 0, 0}, {42, 2, 1}};
    ASSERT_EQ(a.offset, Point(0, 0, 0));
    ASSERT_EQ(a.sizes, Dim(42, 2, 1));
    ASSERT_EQ(a.begin(), Point(0, 0, 0));
    ASSERT_EQ(a.end(), Point(42, 2, 1));
    ASSERT_EQ(a.size(), 84);
    ASSERT_EQ(a.is_empty(), false);

    Range<3> b = {{1, 1, 1}, {3, 2, 1}};
    ASSERT_EQ(b.offset, Point(1, 1, 1));
    ASSERT_EQ(b.sizes, Dim(3, 2, 1));
    ASSERT_EQ(b.begin(), Point(1, 1, 1));
    ASSERT_EQ(b.end(), Point(4, 3, 2));
    ASSERT_EQ(b.size(), 6);
    ASSERT_EQ(b.is_empty(), false);

    Range<3> c = {{1, -5, 1}, {2, 2, 2}};
    ASSERT_EQ(c.offset, Point(1, -5, 1));
    ASSERT_EQ(c.sizes, Dim(2, 2, 2));
    ASSERT_EQ(c.begin(), Point(1, -5, 1));
    ASSERT_EQ(c.end(), Point(3, -3, 3));
    ASSERT_EQ(c.size(), 8);
    ASSERT_EQ(c.is_empty(), false);

    Range<3> d = {{5, 1, 2}, {-5, 3, 1}};
    ASSERT_EQ(d.offset, Point(5, 1, 2));
    ASSERT_EQ(d.sizes, Dim(-5, 3, 1));
    ASSERT_EQ(d.begin(), Point(5, 1, 2));
    ASSERT_EQ(d.end(), Point(5, 4, 3));
    ASSERT_EQ(d.size(), 0);
    ASSERT_EQ(d.is_empty(), true);
}

TEST(Geometry, range_intersection) {
    Range<3> a = {{0, 0, 0}, {42, 2, 5}};
    Range<3> b = {{1, 1, 1}, {3, 1, 1}};
    Range<3> c = {{1, -5, 1}, {2, 20, 2}};
    Range<3> d = {{5, 1, 2}, {-3, 3, 1}};

    ASSERT_EQ(a.intersection(a), a);
    ASSERT_EQ(a.intersection(b), (Range<3>{{1, 1, 1}, {3, 1, 1}}));
    ASSERT_EQ(a.intersection(c), (Range<3>{{1, 0, 1}, {2, 2, 2}}));
    ASSERT_EQ(a.intersection(d),  (Range<3>{{5, 1, 2}, {-3, 1, 1}}));

    ASSERT_EQ(b.intersection(a), (Range<3>{{1, 1, 1}, {3, 1, 1}}));
    ASSERT_EQ(b.intersection(b), b);
    ASSERT_EQ(b.intersection(c), (Range<3>{{1, 1, 1}, {2, 1, 1}}));
    ASSERT_EQ(b.intersection(d), Range<3>());

    ASSERT_EQ(c.intersection(a), (Range<3>{{1, 0, 1}, {2, 2, 2}}));
    ASSERT_EQ(c.intersection(b), (Range<3>{{1, 1, 1}, {2, 1, 1}}));
    ASSERT_EQ(c.intersection(c), c);
    ASSERT_EQ(c.intersection(d), Range<3>());

    ASSERT_EQ(d.intersection(a), (Range<3>{{5, 1, 2}, {-3, 1, 1}}));
    ASSERT_EQ(d.intersection(b), Range<3>());
    ASSERT_EQ(d.intersection(c), Range<3>());
    ASSERT_EQ(d.intersection(d), Range<3>());
    
    
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
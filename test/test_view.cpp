#include <gtest/gtest.h>

#include "kmm/utils/view.hpp"

using namespace kmm;

TEST(View, bound_left_to_right_layout) {
    std::vector<int> vec = {1, 2, 3, 4, 5, 6, 7, 8};
    basic_view<int, views::dynamic_domain<1>, views::left_to_right_layout<1>> v = {vec.data(), {{8}}};

    ASSERT_EQ(v.offset(), 0);
    ASSERT_EQ(v.size(0), 8);
    ASSERT_EQ(v.begin(), 0);
    ASSERT_EQ(v.end(), 8);
    ASSERT_EQ(v.data(), vec.data());
    ASSERT_EQ(v.stride(), 1);
    ASSERT_EQ(v.strides(), 1);
    ASSERT_EQ(v.offsets(), 0);
    ASSERT_EQ(v.sizes(), 8);

    ASSERT_EQ(v.data_at({0}), &vec[0]);
    ASSERT_EQ(v.data_at({4}), &vec[4]);
    ASSERT_EQ(v.data_at({8}), &vec[8]);

    ASSERT_EQ(v.access({0}), vec[0]);
    ASSERT_EQ(v.access({4}), vec[4]);

    ASSERT_EQ(v[0], vec[0]);
    ASSERT_EQ(v[4], vec[4]);
}

TEST(View, bound2_left_to_right_layout) {
    std::vector<int> vec = {1, 2, 3, 4, 5, 6, 7, 8};
    basic_view<int, views::dynamic_domain<2>, views::left_to_right_layout<2>> v = {vec.data(), {{4, 2}}};

    ASSERT_EQ(v.offset(0), 0);
    ASSERT_EQ(v.offset(1), 0);
    ASSERT_EQ(v.size(0), 4);
    ASSERT_EQ(v.size(1), 2);
    ASSERT_EQ(v.begin(0), 0);
    ASSERT_EQ(v.begin(1), 0);
    ASSERT_EQ(v.end(0), 4);
    ASSERT_EQ(v.end(1), 2);
    ASSERT_EQ(v.data(), vec.data());
    ASSERT_EQ(v.stride(0), 1);
    ASSERT_EQ(v.stride(1), 4);
    ASSERT_EQ(v.strides()[0], 1);
    ASSERT_EQ(v.strides()[1], 4);
    ASSERT_EQ(v.offsets()[0], 0);
    ASSERT_EQ(v.offsets()[1], 0);
    ASSERT_EQ(v.sizes()[0], 4);
    ASSERT_EQ(v.sizes()[1], 2);

    ASSERT_EQ(v.data_at({0, 0}), &vec[0]);
    ASSERT_EQ(v.data_at({1, 1}), &vec[5]);
    ASSERT_EQ(v.data_at({3, 1}), &vec[7]);
    ASSERT_EQ(v.data_at({3, 2}), &vec[11]);

    ASSERT_EQ(v.access({0, 1}), vec[4]);
    ASSERT_EQ(v.access({3, 1}), vec[7]);

    ASSERT_EQ(v[0][1], vec[4]);
    ASSERT_EQ(v[3][0], vec[3]);
}

TEST(View, bound2_right_to_left_layout) {
    std::vector<int> vec = {1, 2, 3, 4, 5, 6, 7, 8};
    basic_view<int, views::dynamic_domain<2>, views::right_to_left_layout<2>> v = {vec.data(), {{4, 2}}};

    ASSERT_EQ(v.offset(0), 0);
    ASSERT_EQ(v.offset(1), 0);
    ASSERT_EQ(v.size(0), 4);
    ASSERT_EQ(v.size(1), 2);
    ASSERT_EQ(v.begin(0), 0);
    ASSERT_EQ(v.begin(1), 0);
    ASSERT_EQ(v.end(0), 4);
    ASSERT_EQ(v.end(1), 2);
    ASSERT_EQ(v.data(), vec.data());
    ASSERT_EQ(v.stride(0), 2);
    ASSERT_EQ(v.stride(1), 1);
    ASSERT_EQ(v.strides()[0], 2);
    ASSERT_EQ(v.strides()[1], 1);
    ASSERT_EQ(v.offsets()[0], 0);
    ASSERT_EQ(v.offsets()[1], 0);
    ASSERT_EQ(v.sizes()[0], 4);
    ASSERT_EQ(v.sizes()[1], 2);

    ASSERT_EQ(v.data_at({0, 0}), &vec[0]);
    ASSERT_EQ(v.data_at({1, 1}), &vec[3]);
    ASSERT_EQ(v.data_at({3, 1}), &vec[7]);
    ASSERT_EQ(v.data_at({3, 2}), &vec[8]);

    ASSERT_EQ(v.access({0, 1}), vec[1]);
    ASSERT_EQ(v.access({3, 1}), vec[7]);

    ASSERT_EQ(v[0][1], vec[1]);
    ASSERT_EQ(v[3][0], vec[6]);
}

TEST(View, subbound2_right_to_left_layout) {
    std::vector<int> vec = {1, 2, 3, 4, 5, 6, 7, 8};
    basic_view<int, views::dynamic_subdomain<2>, views::right_to_left_layout<2>> v = {
        vec.data(),
        {{100, 42}, {4, 2}}};

    ASSERT_EQ(v.offset(0), 100);
    ASSERT_EQ(v.offset(1), 42);
    ASSERT_EQ(v.size(0), 4);
    ASSERT_EQ(v.size(1), 2);
    ASSERT_EQ(v.begin(0), 100);
    ASSERT_EQ(v.begin(1), 42);
    ASSERT_EQ(v.end(0), 104);
    ASSERT_EQ(v.end(1), 44);
    ASSERT_EQ(v.data(), vec.data());
    ASSERT_EQ(v.stride(0), 2);
    ASSERT_EQ(v.stride(1), 1);
    ASSERT_EQ(v.strides()[0], 2);
    ASSERT_EQ(v.strides()[1], 1);
    ASSERT_EQ(v.offsets()[0], 100);
    ASSERT_EQ(v.offsets()[1], 42);
    ASSERT_EQ(v.sizes()[0], 4);
    ASSERT_EQ(v.sizes()[1], 2);

    ASSERT_EQ(v.data_at({100, 42}), &vec[0]);
    ASSERT_EQ(v.data_at({101, 43}), &vec[3]);
    ASSERT_EQ(v.data_at({103, 43}), &vec[7]);
    ASSERT_EQ(v.data_at({103, 44}), &vec[8]);

    ASSERT_EQ(v.access({100, 43}), vec[1]);
    ASSERT_EQ(v.access({103, 43}), vec[7]);

    ASSERT_EQ(v[100][43], vec[1]);
    ASSERT_EQ(v[103][42], vec[6]);
}

TEST(View, domain_conversions) {
#define ASSERT_CORRECT_VIEW(p)    \
    ASSERT_EQ((p).offset(0), 0);  \
    ASSERT_EQ((p).offset(1), 0);  \
    ASSERT_EQ((p).size(0), 10);   \
    ASSERT_EQ((p).size(1), 20);   \
    ASSERT_EQ((p).stride(0), 20); \
    ASSERT_EQ((p).stride(1), 1);  \
    ASSERT_TRUE((p).is_contiguous());

    auto a = basic_view<  //
        int,
        views::static_domain<views::default_index_type, 10, 20>,
        views::right_to_left_layout<2>> {nullptr};
    ASSERT_CORRECT_VIEW(a);

    auto b = basic_view<  //
        int,
        views::dynamic_domain<2>,
        views::right_to_left_layout<2>>(a);
    ASSERT_CORRECT_VIEW(b);

    auto c = basic_view<  //
        int,
        views::dynamic_subdomain<2>,
        views::right_to_left_layout<2>>(a);
    ASSERT_CORRECT_VIEW(c);

    auto d = basic_view<  //
        int,
        views::dynamic_subdomain<2>,
        views::right_to_left_layout<2>>(b);
    ASSERT_CORRECT_VIEW(d);

    auto e = basic_view<  //
        int,
        views::dynamic_subdomain<2>,
        views::dynamic_layout<2>>(a);
    ASSERT_CORRECT_VIEW(e);

    auto f = basic_view<  //
        int,
        views::dynamic_subdomain<2>,
        views::dynamic_layout<2>>(b);
    ASSERT_CORRECT_VIEW(f);

    auto g = basic_view<  //
        int,
        views::dynamic_subdomain<2>,
        views::dynamic_layout<2>>(c);
    ASSERT_CORRECT_VIEW(g);

    auto h = basic_view<  //
        int,
        views::dynamic_subdomain<2>,
        views::dynamic_layout<2>>(d);
    ASSERT_CORRECT_VIEW(h);

#undef ASSERT_CORRECT_VIEW
}

TEST(View, subdomain_conversions) {
#define ASSERT_CORRECT_VIEW(p)    \
    ASSERT_EQ((p).offset(0), 3);  \
    ASSERT_EQ((p).offset(1), 7);  \
    ASSERT_EQ((p).size(0), 10);   \
    ASSERT_EQ((p).size(1), 20);   \
    ASSERT_EQ((p).stride(0), 20); \
    ASSERT_EQ((p).stride(1), 1);  \
    ASSERT_TRUE((p).is_contiguous());

    auto a = basic_view<  //
        int,
        views::static_offset<views::static_domain<views::default_index_type, 10, 20>, 3, 7>,
        views::right_to_left_layout<2>> {nullptr};
    ASSERT_CORRECT_VIEW(a);

    auto b = basic_view<  //
        int,
        views::static_offset<views::dynamic_domain<2>, 3, 7>,
        views::right_to_left_layout<2>>(a);
    ASSERT_CORRECT_VIEW(b);

    auto c = basic_view<  //
        int,
        views::dynamic_subdomain<2>,
        views::right_to_left_layout<2>>(a);
    ASSERT_CORRECT_VIEW(c);

    auto d = basic_view<  //
        int,
        views::dynamic_subdomain<2>,
        views::right_to_left_layout<2>>(b);
    ASSERT_CORRECT_VIEW(d);

    auto e = basic_view<  //
        int,
        views::dynamic_subdomain<2>,
        views::dynamic_layout<2>>(a);
    ASSERT_CORRECT_VIEW(e);

    auto f = basic_view<
        int,
        views::dynamic_subdomain<2>,
        views::dynamic_layout<2>>(b);
    ASSERT_CORRECT_VIEW(f);

    auto g = basic_view<
        int,
        views::dynamic_subdomain<2>,
        views::dynamic_layout<2>>(c);
    ASSERT_CORRECT_VIEW(g);

    auto h = basic_view<
        int,
        views::dynamic_subdomain<2>,
        views::dynamic_layout<2>>(d);
    ASSERT_CORRECT_VIEW(h);

#undef ASSERT_CORRECT_VIEW
}

TEST(View, drop_axis_dim2) {
    auto vec = std::vector<float>(200);
    auto a = basic_view<  //
        float,
        views::dynamic_subdomain<2>,
        views::right_to_left_layout<2>> {vec.data(), {{3, 7}, {10, 20}}};

    // Drop axis 0
    basic_view<float, views::dynamic_subdomain<1>, views::right_to_left_layout<1>> b = a.drop_axis();
    ASSERT_EQ(b.size(0), 20);
    ASSERT_EQ(b.offset(0), 7);
    ASSERT_EQ(b.stride(0), 1);
    ASSERT_EQ(b.data(), vec.data());

    b = a.drop_axis(5);
    ASSERT_EQ(b.size(0), 20);
    ASSERT_EQ(b.offset(0), 7);
    ASSERT_EQ(b.stride(0), 1);
    ASSERT_EQ(b.data(), vec.data() + 2 * 20);

    // Drop axis 1
    basic_view<float, views::dynamic_subdomain<1>, views::dynamic_layout<1>> c = a.drop_axis<1>();
    ASSERT_EQ(c.size(0), 10);
    ASSERT_EQ(c.offset(0), 3);
    ASSERT_EQ(c.stride(0), 20);
    ASSERT_EQ(c.data(), vec.data());

    c = a.drop_axis<1>(13);
    ASSERT_EQ(c.size(0), 10);
    ASSERT_EQ(c.offset(0), 3);
    ASSERT_EQ(c.stride(0), 20);
    ASSERT_EQ(c.data(), vec.data() + 6);
}

TEST(View, drop_axis_dim3) {
    auto vec = std::vector<float>(200);
    auto a = basic_view<  //
        float,
        views::dynamic_subdomain<3>,
        views::right_to_left_layout<3>> {
        vec.data(), {{3, 7, 1}, {2, 5, 20}}
    };

    // Drop axis 0
    basic_view<float, views::dynamic_subdomain<2>, views::right_to_left_layout<2>> b = a.drop_axis();
    ASSERT_EQ(b.size(0), 5);
    ASSERT_EQ(b.offset(0), 7);
    ASSERT_EQ(b.stride(0), 20);
    ASSERT_EQ(b.size(1), 20);
    ASSERT_EQ(b.offset(1), 1);
    ASSERT_EQ(b.stride(1), 1);
    ASSERT_EQ(b.data(), vec.data());

    b = a.drop_axis(4);
    ASSERT_EQ(b.size(0), 5);
    ASSERT_EQ(b.offset(0), 7);
    ASSERT_EQ(b.stride(0), 20);
    ASSERT_EQ(b.size(1), 20);
    ASSERT_EQ(b.offset(1), 1);
    ASSERT_EQ(b.stride(1), 1);
    ASSERT_EQ(b.data() - vec.data(), 100);

    // Drop axis 1
    basic_view<float, views::dynamic_subdomain<2>, views::dynamic_layout<2>> c = a.drop_axis<1>();
    ASSERT_EQ(c.size(0), 2);
    ASSERT_EQ(c.offset(0), 3);
    ASSERT_EQ(c.stride(0), 100);
    ASSERT_EQ(c.size(1), 20);
    ASSERT_EQ(c.offset(1), 1);
    ASSERT_EQ(c.stride(1), 1);
    ASSERT_EQ(c.data(), vec.data());

    c = a.drop_axis<1>(9);
    ASSERT_EQ(c.size(0), 2);
    ASSERT_EQ(c.offset(0), 3);
    ASSERT_EQ(c.stride(0), 100);
    ASSERT_EQ(c.size(1), 20);
    ASSERT_EQ(c.offset(1), 1);
    ASSERT_EQ(c.stride(1), 1);
    ASSERT_EQ(c.data() - vec.data(),  + 40);

    // Drop axis 3
    basic_view<float, views::dynamic_subdomain<2>, views::dynamic_layout<2>> d = a.drop_axis<2>();
    ASSERT_EQ(d.size(0), 2);
    ASSERT_EQ(d.offset(0), 3);
    ASSERT_EQ(d.stride(0), 100);
    ASSERT_EQ(d.size(1), 5);
    ASSERT_EQ(d.offset(1), 7);
    ASSERT_EQ(d.stride(1), 20);
    ASSERT_EQ(d.data(), vec.data());

    d = a.drop_axis<2>(13);
    ASSERT_EQ(d.size(0), 2);
    ASSERT_EQ(d.offset(0), 3);
    ASSERT_EQ(d.stride(0), 100);
    ASSERT_EQ(d.size(1), 5);
    ASSERT_EQ(d.offset(1), 7);
    ASSERT_EQ(d.stride(1), 20);
    ASSERT_EQ(d.data() - vec.data(), 12);
}

TEST(View, scalar) {
    auto value = int(1);
    auto v = basic_view<  //
        int,
        views::dynamic_subdomain<0>,
        views::right_to_left_layout<0>> {&value};

    ASSERT_EQ(v.data(), &value);
    ASSERT_EQ(v.data_at({}), &value);
    ASSERT_EQ(v.access({}), value);

    *v = 2;
    ASSERT_EQ(value, 2);
}
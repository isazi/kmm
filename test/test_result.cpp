#include <gtest/gtest.h>

#include "kmm/utils/result.hpp"

using namespace kmm;

TEST(ErrorPtrTest, emtpy) {
    ErrorPtr x;

    ASSERT_FALSE(x.has_value());
    ASSERT_FALSE(x);
    ASSERT_THROW(x.rethrow(), EmptyException);
    ASSERT_NO_THROW(x.rethrow_if_present());
    ASSERT_EQ(x.what(), "unknown exception");
    ASSERT_EQ(x.get_if<std::exception>(), std::nullopt);
    ASSERT_FALSE(x.is<std::exception>());
    ASSERT_EQ(x.type(), typeid(nullptr));
    ASSERT_NE(x.type(), typeid(std::runtime_error));
    ASSERT_NE(x.type(), typeid(std::exception));
    ASSERT_FALSE(x.get_exception_ptr());
    ASSERT_EQ(x, ErrorPtr());
    ASSERT_EQ(x, x);
}

TEST(ErrorPtrTest, from_current_exception) {
    ErrorPtr x;

    try {
        throw std::runtime_error("foo");
    } catch (...) {
        x = ErrorPtr::from_current_exception();
    }

    ASSERT_TRUE(x.has_value());
    ASSERT_TRUE(x);
    ASSERT_THROW(x.rethrow(), std::runtime_error);
    ASSERT_THROW(x.rethrow_if_present(), std::runtime_error);
    ASSERT_EQ(x.what(), "foo");
    ASSERT_TRUE(x.get_if<std::exception>());
    ASSERT_TRUE(x.get_if<std::runtime_error>());
    ASSERT_TRUE(x.is<std::exception>());
    ASSERT_TRUE(x.is<std::runtime_error>());
    ASSERT_FALSE(x.is<std::logic_error>());
    ASSERT_NE(x.type(), typeid(nullptr));
    ASSERT_EQ(x.type(), typeid(std::runtime_error));
    ASSERT_NE(x.type(), typeid(std::exception));
    ASSERT_TRUE(x.get_exception_ptr());
    ASSERT_NE(x, ErrorPtr());
    ASSERT_EQ(x, x);
}

TEST(ErrorPtrTest, constructors) {
    ErrorPtr empty;
    ASSERT_EQ(empty.what(), "unknown exception");

    ErrorPtr a = ErrorPtr(std::make_exception_ptr(std::runtime_error("a")));
    ASSERT_EQ(a.what(), "a");

    ErrorPtr b = ErrorPtr(std::runtime_error("b"));
    ASSERT_EQ(b.what(), "b");

    ErrorPtr c = ErrorPtr("c");
    ASSERT_EQ(c.what(), "c");

    ErrorPtr d = ErrorPtr(std::string("d"));
    ASSERT_EQ(d.what(), "d");

    ErrorPtr e = ErrorPtr::from_exception(std::runtime_error("e"));
    ASSERT_EQ(e.what(), "e");

    ErrorPtr f;
    try {
        throw std::runtime_error("f");
    } catch (...) {
        f = ErrorPtr::from_current_exception();
    }
    ASSERT_EQ(f.what(), "f");
}

TEST(ResultTest, nonvoid_value) {
    Result<int> x = {123};

    ASSERT_TRUE(x.has_value());
    ASSERT_TRUE(x);
    ASSERT_TRUE(x.value_if_present());
    ASSERT_EQ(*x.value_if_present(), 123);
    ASSERT_EQ(x.value(), 123);
    ASSERT_EQ(*x, 123);
    ASSERT_FALSE(x.has_error());
    ASSERT_THROW(x.rethrow_error(), EmptyException);
    ASSERT_NO_THROW(x.rethrow_if_error());
    ASSERT_THROW(x.error(), EmptyException);
    ASSERT_FALSE(x.error_if_present());
    ASSERT_EQ(x, Result<int> {123});
    ASSERT_NE(x, Result<int> {456});
    ASSERT_NE(x, Result<int>::from_error("abc"));
}

TEST(ResultTest, nonvoid_error) {
    ErrorPtr error = ErrorPtr::from_exception(std::runtime_error("foo"));
    Result<int> x = error;

    ASSERT_FALSE(x.has_value());
    ASSERT_FALSE(x);
    ASSERT_FALSE(x.value_if_present());
    ASSERT_THROW(x.value(), std::runtime_error);
    ASSERT_THROW(*x, std::runtime_error);
    ASSERT_TRUE(x.has_error());
    ASSERT_THROW(x.rethrow_error(), std::runtime_error);
    ASSERT_THROW(x.rethrow_if_error(), std::runtime_error);
    ASSERT_EQ(x.error().what(), "foo");
    ASSERT_TRUE(x.error_if_present());
    ASSERT_NE(x, Result<int> {123});
    ASSERT_NE(x, Result<int> {456});
    ASSERT_NE(x, Result<int>::from_error("abc"));
    ASSERT_EQ(x, Result<int> {error});
}

TEST(ResultTest, void_value) {
    Result<void> x;

    ASSERT_TRUE(x.has_value());
    ASSERT_TRUE(x);
    ASSERT_NO_THROW(x.value());
    ASSERT_FALSE(x.has_error());
    ASSERT_THROW(x.rethrow_error(), EmptyException);
    ASSERT_NO_THROW(x.rethrow_if_error());
    ASSERT_THROW(x.error(), EmptyException);
    ASSERT_FALSE(x.error_if_present());
    ASSERT_EQ(x, Result<void> {});
    ASSERT_NE(x, Result<void>::from_error("abc"));
}

TEST(ResultTest, void_error) {
    ErrorPtr error = ErrorPtr::from_exception(std::runtime_error("foo"));
    Result<void> x = error;

    ASSERT_FALSE(x.has_value());
    ASSERT_FALSE(x);
    ASSERT_THROW(x.value(), std::runtime_error);
    ASSERT_TRUE(x.has_error());
    ASSERT_THROW(x.rethrow_error(), std::runtime_error);
    ASSERT_THROW(x.rethrow_if_error(), std::runtime_error);
    ASSERT_EQ(x.error().what(), "foo");
    ASSERT_TRUE(x.error_if_present());
    ASSERT_NE(x, Result<void> {});
    ASSERT_NE(x, Result<void>::from_error("abc"));
    ASSERT_EQ(x, Result<void> {error});
}

TEST(ResultTest, try_catch_int) {
    auto x = try_catch([&] { return 123; });

    ASSERT_TRUE((std::is_same_v<decltype(x), Result<int>>));
    ASSERT_EQ(x.value(), 123);

    auto y = try_catch([&] {
        if (1 + 1 == 2) {
            throw std::runtime_error("foo");
        }

        return 123;
    });

    ASSERT_TRUE((std::is_same_v<decltype(y), Result<int>>));
    ASSERT_EQ(y.error().what(), "foo");
}

TEST(ResultTest, try_catch_void) {
    auto x = try_catch([&] {});

    ASSERT_TRUE((std::is_same_v<decltype(x), Result<void>>));
    ASSERT_NO_THROW(x.value());

    auto y = try_catch([&] { throw std::runtime_error("foo"); });

    ASSERT_TRUE((std::is_same_v<decltype(y), Result<void>>));
    ASSERT_EQ(y.error().what(), "foo");
}
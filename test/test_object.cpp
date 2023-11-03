#include <gtest/gtest.h>

#include "kmm/object.hpp"

using namespace kmm;

TEST(ObjectTest, int) {
    ObjectHandle obj = make_object(int(5));

    ASSERT_EQ(obj->type_info(), typeid(int));
    ASSERT_EQ(obj->type_name(), "i");
    ASSERT_NE(obj->get_if<int>(), nullptr);
    ASSERT_EQ(obj->is<int>(), true);
    ASSERT_EQ(obj->get<int>(), 5);
    ASSERT_EQ(*object_cast<int>(obj), 5);

    ASSERT_EQ(obj->is<float>(), false);
    ASSERT_EQ(obj->get_if<float>(), nullptr);
    ASSERT_ANY_THROW(obj->get<float>());
    ASSERT_ANY_THROW(object_cast<float>(obj));
}

struct Custom {};

TEST(ObjectTest, Custom) {
    ObjectHandle obj = make_object(Custom {});

    ASSERT_EQ(obj->type_info(), typeid(Custom));
    ASSERT_EQ(obj->type_name(), "6Custom");
    ASSERT_EQ(obj->get_if<int>(), nullptr);
    ASSERT_NE(obj->get_if<Custom>(), nullptr);
    ASSERT_NO_THROW(obj->get<Custom>());

    ASSERT_EQ(obj->is<int>(), false);
    ASSERT_EQ(obj->get_if<int>(), nullptr);
    ASSERT_ANY_THROW(obj->get<int>());
    ASSERT_ANY_THROW(object_cast<int>(obj));
}
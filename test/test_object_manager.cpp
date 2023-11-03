#include <gtest/gtest.h>

#include "kmm/object_manager.hpp"

using namespace kmm;

TEST(ObjectManager, basic) {
    auto manager = ObjectManager();

    // Create 4 and 5
    manager.create_object(ObjectId(4), make_object(123));
    manager.create_object(ObjectId(5), make_object(456));

    ASSERT_EQ(manager.get_object(ObjectId(4))->get<int>(), 123);
    ASSERT_EQ(manager.get_object(ObjectId(5))->get<int>(), 456);

    // Overwrite 4
    manager.create_object(ObjectId(4), make_object(999));

    ASSERT_EQ(manager.get_object(ObjectId(4))->get<int>(), 999);
    ASSERT_EQ(manager.get_object(ObjectId(5))->get<int>(), 456);

    // Delete 4
    manager.delete_object(ObjectId(4));
    ASSERT_ANY_THROW(manager.get_object(ObjectId(4)));
    ASSERT_EQ(manager.get_object(ObjectId(5))->get<int>(), 456);
}

TEST(ObjectManager, poison) {
    auto manager = ObjectManager();

    // Poison 4
    manager.poison_object(ObjectId(4), std::string("some error"));
    ASSERT_ANY_THROW(
        try { manager.get_object(ObjectId(4)); } catch (const std::exception& e) {
            ASSERT_EQ(e.what(), std::string("object is poisoned: some error"));
            throw;
        });
    manager.delete_object(ObjectId(4));

    // Create 5 and then poison it
    manager.create_object(ObjectId(5), make_object(456));
    ASSERT_NO_THROW(manager.get_object(ObjectId(5)));
    manager.poison_object(ObjectId(5), std::string("another error"));
    ASSERT_ANY_THROW(
        try { manager.get_object(ObjectId(5)); } catch (const std::exception& e) {
            ASSERT_EQ(e.what(), std::string("object is poisoned: another error"));
            throw;
        });
    manager.delete_object(ObjectId(5));
}
#include <gtest/gtest.h>

#include "kmm/memory_manager.hpp"

class MockWaker: public kmm::Waker {
    void trigger_wakeup() const override {}
};

class MockAllocation: public kmm::MemoryAllocation {
  public:
    MockAllocation(int id, kmm::DeviceId device_id, size_t num_bytes) :
        id(id),
        num_bytes(num_bytes),
        device_id(device_id) {}

    int id;
    size_t num_bytes;
    kmm::DeviceId device_id;
};

class MockMemory: public kmm::Memory {
  public:
    std::optional<std::unique_ptr<kmm::MemoryAllocation>> allocate(
        kmm::DeviceId device_id,
        size_t num_bytes) {
        int id = next_id++;

        if (used[device_id] + num_bytes > 100) {
            return std::nullopt;
        }

        allocations.insert({id, device_id});
        used[device_id] += num_bytes;
        return std::make_unique<MockAllocation>(id++, device_id, num_bytes);
    }

    void deallocate(kmm::DeviceId device_id, std::unique_ptr<kmm::MemoryAllocation> allocation)
        override {
        auto alloc = dynamic_cast<const MockAllocation&>(*allocation);
        ASSERT_EQ(device_id, allocations.at(alloc.id));

        for (const auto& [a, b, _] : transfers) {
            ASSERT_NE(a, alloc.id);
            ASSERT_NE(b, alloc.id);
        }

        used[device_id] -= alloc.num_bytes;
        allocations.erase(alloc.id);
    }

    void copy_async(
        kmm::DeviceId src_id,
        const kmm::MemoryAllocation* src_alloc,
        size_t src_offset,
        kmm::DeviceId dst_id,
        const kmm::MemoryAllocation* dst_alloc,
        size_t dst_offset,
        size_t num_bytes,
        std::unique_ptr<kmm::TransferCompletion> completion) override {
        auto src = dynamic_cast<const MockAllocation&>(*src_alloc);
        ASSERT_EQ(src.device_id, allocations.at(src.id));
        ASSERT_EQ(src.num_bytes, num_bytes);
        ASSERT_EQ(src_offset, 0);

        auto dst = dynamic_cast<const MockAllocation&>(*dst_alloc);
        ASSERT_EQ(dst.device_id, allocations.at(dst.id));
        ASSERT_EQ(dst.num_bytes, num_bytes);
        ASSERT_EQ(dst_offset, 0);

        transfers.emplace_back(src.id, dst.id, std::move(completion));
    }

    bool is_copy_possible(kmm::DeviceId src_id, kmm::DeviceId dst_id) override {
        return true;
    }

    void complete_next_transfer(kmm::DeviceId src_device, kmm::DeviceId dst_device) {
        auto& [src_id, dst_id, callback] = transfers.at(0);

        ASSERT_EQ(allocations.at(src_id), src_device);
        ASSERT_EQ(allocations.at(dst_id), dst_device);
        ASSERT_TRUE(callback);

        callback->mark_job_complete();
        transfers.pop_front();
    }

    std::deque<std::tuple<int, int, std::unique_ptr<kmm::TransferCompletion>>> transfers;
    std::unordered_map<int, kmm::DeviceId> allocations;
    std::unordered_map<kmm::DeviceId, size_t> used;
    int next_id = 1;
};

class MemoryManagerTest: public testing::Test {
  protected:
    void SetUp() override {
        memory = std::make_shared<MockMemory>();
        manager = std::make_shared<kmm::MemoryManager>(memory);
    }

    kmm::BlockLayout build_layout(size_t num_bytes = 1) const {
        return {.num_bytes = num_bytes, .alignment = 1};
    }

    void TearDown() override {
        ASSERT_EQ(memory->allocations.size(), 0);
        ASSERT_EQ(memory->transfers.size(), 0);
    }

    std::shared_ptr<MockMemory> memory;
    std::shared_ptr<kmm::MemoryManager> manager;
};

TEST_F(MemoryManagerTest, basic) {
    auto buffer_id = kmm::BufferId(42);
    auto layout = build_layout();

    auto waker = std::make_shared<MockWaker>();

    manager->create_buffer(buffer_id, layout);

    auto request = manager->create_request(buffer_id, kmm::DeviceId(2), false, waker);
    ASSERT_EQ(manager->poll_request(request), kmm::PollResult::Ready);
    manager->view_buffer(request);
    manager->delete_request(request);

    manager->delete_buffer(buffer_id);
}

TEST_F(MemoryManagerTest, write_then_read) {
    auto buffer_id = kmm::BufferId(42);
    auto layout = build_layout();

    auto waker = std::make_shared<MockWaker>();

    manager->create_buffer(buffer_id, layout);

    auto request = manager->create_request(buffer_id, kmm::DeviceId(2), true, waker);
    ASSERT_EQ(manager->poll_request(request), kmm::PollResult::Ready);
    manager->view_buffer(request);
    manager->delete_request(request);

    request = manager->create_request(buffer_id, kmm::DeviceId(1), false, waker);
    ASSERT_EQ(manager->poll_request(request), kmm::PollResult::Pending);
    memory->complete_next_transfer(kmm::DeviceId(2), kmm::DeviceId(1));
    ASSERT_EQ(manager->poll_request(request), kmm::PollResult::Ready);
    manager->view_buffer(request);
    manager->delete_request(request);

    request = manager->create_request(buffer_id, kmm::DeviceId(1), true, waker);
    ASSERT_EQ(manager->poll_request(request), kmm::PollResult::Ready);
    manager->view_buffer(request);
    manager->delete_request(request);
}

TEST_F(MemoryManagerTest, write_read_write_read) {
    auto buffer_id = kmm::BufferId(42);
    auto layout = build_layout();

    auto waker = std::make_shared<MockWaker>();

    manager->create_buffer(buffer_id, layout);

    auto request = manager->create_request(buffer_id, kmm::DeviceId(2), true, waker);
    ASSERT_EQ(manager->poll_request(request), kmm::PollResult::Ready);
    manager->delete_request(request);

    request = manager->create_request(buffer_id, kmm::DeviceId(1), false, waker);
    ASSERT_EQ(manager->poll_request(request), kmm::PollResult::Pending);
    memory->complete_next_transfer(kmm::DeviceId(2), kmm::DeviceId(1));
    ASSERT_EQ(manager->poll_request(request), kmm::PollResult::Ready);
    manager->delete_request(request);

    request = manager->create_request(buffer_id, kmm::DeviceId(1), true, waker);
    ASSERT_EQ(manager->poll_request(request), kmm::PollResult::Ready);
    manager->delete_request(request);

    request = manager->create_request(buffer_id, kmm::DeviceId(2), false, waker);
    ASSERT_EQ(manager->poll_request(request), kmm::PollResult::Pending);
    memory->complete_next_transfer(kmm::DeviceId(1), kmm::DeviceId(2));
    ASSERT_EQ(manager->poll_request(request), kmm::PollResult::Ready);
    manager->delete_request(request);

    manager->delete_buffer(buffer_id);
}
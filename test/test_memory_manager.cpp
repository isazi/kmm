#include <gtest/gtest.h>

#include "kmm/worker/memory_manager.hpp"

using namespace kmm;

class MockWaker: public Waker {
    void trigger_wakeup(bool allow_progress) const override {}
};

class MockAllocation: public MemoryAllocation {
  public:
    MockAllocation(int id, MemoryId memory_id, size_t num_bytes) :
        id(id),
        num_bytes(num_bytes),
        memory_id(memory_id) {}

    int id;
    size_t num_bytes;
    MemoryId memory_id;
};

class MemoryState {
  public:
    void complete_next_transfer(MemoryId src_device, MemoryId dst_device) {
        auto [src_id, dst_id, callback] = std::move(transfers.at(0));

        ASSERT_EQ(allocations.at(src_id), src_device);
        ASSERT_EQ(allocations.at(dst_id), dst_device);

        transfers.pop_front();
        callback.complete(Result<void>());
    }

    std::deque<std::tuple<int, int, Completion>> transfers;
    std::unordered_map<int, MemoryId> allocations;
    std::unordered_map<MemoryId, size_t> used;
    int next_id = 1;
};

class MockMemory: public Memory {
  public:
    static constexpr size_t HOST_CAPACITY = 1000;
    static constexpr size_t DEVICE_CAPACITY = 100;

    explicit MockMemory(std::shared_ptr<MemoryState> state) : state(state) {}

    std::optional<std::unique_ptr<MemoryAllocation>> allocate(
        MemoryId memory_id,
        size_t num_bytes) {
        size_t capacity = memory_id == MemoryManager::HOST_MEMORY ? HOST_CAPACITY : DEVICE_CAPACITY;
        int id = state->next_id++;

        if (state->used[memory_id] + num_bytes > capacity) {
            return std::nullopt;
        }

        state->allocations.insert({id, memory_id});
        state->used[memory_id] += num_bytes;
        return std::make_unique<MockAllocation>(id++, memory_id, num_bytes);
    }

    void deallocate(MemoryId memory_id, std::unique_ptr<MemoryAllocation> allocation) override {
        auto alloc = dynamic_cast<const MockAllocation&>(*allocation);
        ASSERT_EQ(memory_id, state->allocations.at(alloc.id));

        for (const auto& [a, b, _] : state->transfers) {
            ASSERT_NE(a, alloc.id);
            ASSERT_NE(b, alloc.id);
        }

        state->used[memory_id] -= alloc.num_bytes;
        state->allocations.erase(alloc.id);
    }

    void copy_async(
        MemoryId src_id,
        const MemoryAllocation* src_alloc,
        size_t src_offset,
        MemoryId dst_id,
        const MemoryAllocation* dst_alloc,
        size_t dst_offset,
        size_t num_bytes,
        Completion completion) override {
        auto src = dynamic_cast<const MockAllocation&>(*src_alloc);
        ASSERT_EQ(src.memory_id, state->allocations.at(src.id));
        ASSERT_EQ(src.num_bytes, num_bytes);
        ASSERT_EQ(src_offset, 0);

        auto dst = dynamic_cast<const MockAllocation&>(*dst_alloc);
        ASSERT_EQ(dst.memory_id, state->allocations.at(dst.id));
        ASSERT_EQ(dst.num_bytes, num_bytes);
        ASSERT_EQ(dst_offset, 0);

        state->transfers.emplace_back(src.id, dst.id, std::move(completion));
    }

    void fill_async(
        MemoryId dst_id,
        const MemoryAllocation* dst_alloc,
        size_t dst_offset,
        size_t num_bytes,
        std::vector<uint8_t> fill_bytes,
        Completion completion) {
        auto dst = dynamic_cast<const MockAllocation&>(*dst_alloc);
        ASSERT_EQ(dst.memory_id, state->allocations.at(dst.id));
        ASSERT_EQ(dst.num_bytes, num_bytes);
        ASSERT_EQ(dst_offset, 0);

        state->transfers.emplace_back(dst.id, dst.id, std::move(completion));
    }

    bool is_copy_possible(MemoryId src_id, MemoryId dst_id) override {
        return true;
    }

    std::shared_ptr<MemoryState> state;
};

class MemoryManagerTest: public testing::Test {
  protected:
    void SetUp() override {
        memory = std::make_shared<MemoryState>();
        manager = std::make_shared<MemoryManager>(std::make_unique<MockMemory>(memory));
    }

    BlockLayout make_layout(size_t num_bytes = 1) const {
        return {.num_bytes = num_bytes, .alignment = 1};
    }

    std::shared_ptr<MemoryManager::Transaction> make_transaction() const {
        return manager->create_transaction(std::make_shared<MockWaker>());
    }

    void TearDown() override {
        ASSERT_EQ(memory->allocations.size(), 0);
        ASSERT_EQ(memory->transfers.size(), 0);
    }

    std::shared_ptr<MemoryState> memory;
    std::shared_ptr<MemoryManager> manager;
};

TEST_F(MemoryManagerTest, basic) {
    auto buffer_id = manager->create_buffer(make_layout(50));
    auto transaction = make_transaction();
    auto req = manager->create_request(buffer_id, MemoryId(0), AccessMode::Read, transaction);

    ASSERT_EQ(manager->poll_request(req), PollResult::Ready);
    manager->view_buffer(req);
    manager->delete_request(req);
    manager->delete_buffer(buffer_id);
}

/**
 * Create two requests and check if they are granted simultaneously are not. This is only
 * allowed for `Atomic` and `Read` requests.
 */
TEST_F(MemoryManagerTest, check_access) {
    std::tuple<AccessMode, AccessMode> combinations[] = {
        {AccessMode::Read, AccessMode::Read},
        {AccessMode::Read, AccessMode::ReadWrite},
        {AccessMode::Read, AccessMode::Write},
        {AccessMode::ReadWrite, AccessMode::Read},
        {AccessMode::ReadWrite, AccessMode::ReadWrite},
        {AccessMode::ReadWrite, AccessMode::Write},
        {AccessMode::Write, AccessMode::Read},
        {AccessMode::Write, AccessMode::ReadWrite},
        {AccessMode::Write, AccessMode::Write},
    };

    for (auto [access_a, access_b] : combinations) {
        bool allow_concurrent = (access_a == AccessMode::Read && access_b == AccessMode::Read)
            || (access_a == AccessMode::Write && access_b == AccessMode::Write);

        auto buffer_id = manager->create_buffer(make_layout(50));
        auto transaction = make_transaction();

        auto req_a = manager->create_request(buffer_id, MemoryId(0), access_a, transaction);
        ASSERT_EQ(manager->poll_request(req_a), PollResult::Ready);

        auto req_b = manager->create_request(buffer_id, MemoryId(0), access_b, transaction);
        if (allow_concurrent) {
            ASSERT_EQ(manager->poll_request(req_b), PollResult::Ready);
        } else {
            ASSERT_EQ(manager->poll_request(req_b), PollResult::Pending);
        }

        manager->view_buffer(req_a);
        manager->delete_request(req_a);

        ASSERT_EQ(manager->poll_request(req_b), PollResult::Ready);
        manager->view_buffer(req_b);
        manager->delete_request(req_b);

        manager->delete_buffer(buffer_id);
    }
}

/**
 * Write on memory 0 and then read the result on memory 1. There should be a transfer inbetween.
 * Repeat this 5 times to ensure that the process is repeatable.
 */
TEST_F(MemoryManagerTest, write_transfer_read) {
    auto buffer_id = manager->create_buffer(make_layout(50));
    auto transaction = make_transaction();

    for (size_t repeat = 0; repeat < 5; repeat++) {
        auto req_write =
            manager->create_request(buffer_id, MemoryId(0), AccessMode::ReadWrite, transaction);
        ASSERT_EQ(manager->poll_request(req_write), PollResult::Ready);

        auto req_read =
            manager->create_request(buffer_id, MemoryId(1), AccessMode::Read, transaction);
        ASSERT_EQ(manager->poll_request(req_read), PollResult::Pending);

        ASSERT_EQ(manager->poll_request(req_write), PollResult::Ready);
        manager->view_buffer(req_write);
        manager->delete_request(req_write);

        ASSERT_EQ(manager->poll_request(req_read), PollResult::Pending);
        memory->complete_next_transfer(MemoryId(0), MemoryId(1));

        ASSERT_EQ(manager->poll_request(req_read), PollResult::Ready);
        manager->view_buffer(req_read);
        manager->delete_request(req_read);
    }

    manager->delete_buffer(buffer_id);
}

/**
 * Create two requests that together exceed the memory capacity of the device. The first
 * request should be first granted and the second request should only be granted after the first
 * one finishes.
 */
TEST_F(MemoryManagerTest, cache_eviction) {
    auto buffer_a = manager->create_buffer(make_layout(51));
    auto buffer_b = manager->create_buffer(make_layout(52));

    auto req_a =
        manager->create_request(buffer_a, MemoryId(1), AccessMode::Read, make_transaction());
    ASSERT_EQ(manager->poll_request(req_a), PollResult::Ready);

    auto req_b =
        manager->create_request(buffer_b, MemoryId(1), AccessMode::Read, make_transaction());
    ASSERT_EQ(manager->poll_request(req_b), PollResult::Pending);

    ASSERT_EQ(manager->poll_request(req_a), PollResult::Ready);
    manager->view_buffer(req_a);
    manager->delete_request(req_a);

    ASSERT_EQ(manager->poll_request(req_b), PollResult::Pending);
    memory->complete_next_transfer(MemoryId(1), MemoryId(0));

    ASSERT_EQ(manager->poll_request(req_b), PollResult::Ready);
    manager->view_buffer(req_b);
    manager->delete_request(req_b);

    manager->delete_buffer(buffer_a);
    manager->delete_buffer(buffer_b);
}

/**
 * Create a transaction where the total set of requested buffers exceeds memory capacity. This
 * cannot be granted, but only when all previous requests have finished.
 */
TEST_F(MemoryManagerTest, out_of_memory) {
    auto buffer_a = manager->create_buffer(make_layout(51));
    auto buffer_b = manager->create_buffer(make_layout(52));

    auto req_x =
        manager->create_request(buffer_b, MemoryId(1), AccessMode::Read, make_transaction());
    ASSERT_EQ(manager->poll_request(req_x), PollResult::Ready);

    auto tran = make_transaction();
    auto req_a = manager->create_request(buffer_a, MemoryId(1), AccessMode::Read, tran);
    auto req_b = manager->create_request(buffer_b, MemoryId(1), AccessMode::Read, tran);
    ASSERT_EQ(manager->poll_request(req_a), PollResult::Pending);
    ASSERT_EQ(manager->poll_request(req_b), PollResult::Pending);

    ASSERT_EQ(manager->poll_request(req_x), PollResult::Ready);
    manager->view_buffer(req_x);
    manager->delete_request(req_x);

    ASSERT_EQ(manager->poll_request(req_a), PollResult::Pending);
    ASSERT_EQ(manager->poll_request(req_b), PollResult::Pending);
    memory->complete_next_transfer(MemoryId(1), MemoryId(0));

    ASSERT_EQ(manager->poll_request(req_a), PollResult::Ready);
    ASSERT_EQ(manager->poll_request(req_b), PollResult::Ready);
    manager->view_buffer(req_a);
    //    manager->view_buffer(req_b);
    manager->delete_request(req_a);
    manager->delete_request(req_b);

    manager->delete_buffer(buffer_a);
    manager->delete_buffer(buffer_b);
}
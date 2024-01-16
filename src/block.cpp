#include "kmm/block.hpp"
#include "kmm/runtime.hpp"
#include "kmm/runtime_impl.hpp"

namespace kmm {

Block::Block(std::shared_ptr<RuntimeImpl> runtime, BlockId id) :
    m_id(id),
    m_runtime(std::move(runtime)) {
    KMM_ASSERT(m_runtime != nullptr);
}

Block::~Block() {
    if (m_id != BlockId::invalid()) {
        m_runtime->delete_block(m_id);
    }
}

std::shared_ptr<Block> Block::create(
    std::shared_ptr<RuntimeImpl> runtime,
    std::unique_ptr<BlockHeader> header,
    const void* data_ptr,
    size_t num_bytes,
    MemoryId memory_id) {
    auto block_id = runtime->create_block(memory_id, std::move(header), data_ptr, num_bytes);
    return std::make_shared<Block>(std::move(runtime), block_id);
}

Runtime Block::runtime() const {
    return m_runtime;
}

EventId Block::prefetch(MemoryId memory_id, EventList dependencies) const {
    return m_runtime->submit_block_prefetch(m_id, memory_id, std::move(dependencies));
}

EventId Block::submit_barrier() const {
    return m_runtime->submit_block_barrier(m_id);
}

void Block::synchronize() const {
    m_runtime->query_event(submit_barrier(), std::chrono::system_clock::time_point::max());
}

BlockId Block::release() {
    return std::exchange(m_id, BlockId::invalid());
}

std::shared_ptr<BlockHeader> Block::header() const {
    return m_runtime->read_block_header(m_id);
}

std::shared_ptr<BlockHeader> Block::read(void* dst_ptr, size_t nbytes) const {
    return m_runtime->read_block(m_id, dst_ptr, nbytes);
}

}  // namespace kmm
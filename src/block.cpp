#include "kmm/block.hpp"
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
}  // namespace kmm
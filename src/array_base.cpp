#include "kmm/array_base.hpp"
#include "kmm/runtime.hpp"
#include "kmm/runtime_handle.hpp"

namespace kmm {

bool ArrayBase::has_block() const {
    return bool(m_block);
}

std::shared_ptr<Block> ArrayBase::block() const {
    KMM_ASSERT(m_block != nullptr);
    return m_block;
}

BlockId ArrayBase::id() const {
    return block()->id();
}

RuntimeHandle ArrayBase::runtime() const {
    return {m_block->runtime().shared_from_this()};
}

void ArrayBase::synchronize() const {
    if (m_block) {
        auto event_id = m_block->runtime().submit_block_barrier(m_block->id());
        m_block->runtime().query_event(event_id, std::chrono::system_clock::time_point::max());
    }
}

EventId ArrayBase::prefetch(MemoryId memory_id, EventList dependencies) const {
    if (m_block) {
        return m_block->runtime().submit_block_prefetch(
            m_block->id(),
            memory_id,
            std::move(dependencies));
    } else {
        return EventId::invalid();
    }
}

void ArrayBase::read_bytes(void* dst_ptr, size_t nbytes) const {
    block()->runtime().read_block(m_block->id(), dst_ptr, nbytes);
}

index_t ArrayBase::size() const {
    index_t n = 1;

    for (size_t i = 0; i < rank(); i++) {
        n = checked_mul(n, size(i));
    }

    return n;
}

bool ArrayBase::is_empty() const {
    for (size_t i = 0; i < rank(); i++) {
        if (size(i) == 0) {
            return true;
        }
    }

    return false;
}

Block::Block(std::shared_ptr<Runtime> runtime, BlockId id) :
    m_id(id),
    m_runtime(std::move(runtime)) {
    KMM_ASSERT(m_runtime != nullptr);
}

Block::~Block() {
    if (m_id != BlockId::invalid()) {
        m_runtime->delete_block(m_id);
    }
}

}  // namespace kmm
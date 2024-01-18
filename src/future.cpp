#include "kmm/future.hpp"

namespace kmm {

bool FutureBase::has_block() const {
    return bool(m_block);
}

std::shared_ptr<Block> FutureBase::block() const {
    if (!m_block) {
        throw std::runtime_error("cannot access future that has not been initialized yet");
    }

    return m_block;
}

BlockId FutureBase::id() const {
    return block()->id();
}

Runtime FutureBase::runtime() const {
    return block()->runtime();
}

bool FutureBase::is(const std::type_info& that) const {
    return m_type != nullptr && *m_type == that;
}

void FutureBase::synchronize() const {
    if (m_block) {
        m_block->synchronize();
    }
}

}  // namespace kmm
#include "kmm/future.hpp"

namespace kmm {

bool FutureAny::has_block() const {
    return bool(m_block);
}

std::shared_ptr<Block> FutureAny::block() const {
    if (!m_block) {
        throw std::runtime_error("cannot access future that has not been initialized yet");
    }

    return m_block;
}

BlockId FutureAny::id() const {
    return block()->id();
}

Runtime FutureAny::runtime() const {
    return block()->runtime();
}

bool FutureAny::is(const std::type_info& that) const {
    return m_type != nullptr && *m_type == that;
}

void FutureAny::synchronize() const {
    if (m_block) {
        m_block->synchronize();
    }
}

}  // namespace kmm
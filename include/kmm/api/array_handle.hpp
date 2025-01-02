#pragma once

#include "kmm/core/geometry.hpp"
#include "kmm/core/identifiers.hpp"

namespace kmm {

class Worker;

template<size_t N>
class ArrayHandle: public std::enable_shared_from_this<ArrayHandle<N>>, public DataDistribution<N> {
    KMM_NOT_COPYABLE_OR_MOVABLE(ArrayHandle)

  public:
    ArrayHandle(Worker& worker, DataDistribution<N> distribution);
    ~ArrayHandle();

    void copy_bytes(void* dest_addr, size_t element_size) const;
    void synchronize() const;

    const Worker& worker() const {
        return *m_worker;
    }

  private:
    std::shared_ptr<Worker> m_worker;
};

}  // namespace kmm
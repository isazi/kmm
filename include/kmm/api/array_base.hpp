#include <memory>
#include <vector>

#include "kmm/api/runtime_impl.hpp"
#include "kmm/core/geometry.hpp"

namespace kmm {

class ArrayBase {
  public:
    virtual ~ArrayBase() = default;
    virtual size_t rank() const = 0;
    virtual size_t size(size_t axis) const = 0;
    virtual size_t chunk_size(size_t axis) const = 0;
    virtual size_t num_chunks() const = 0;
    virtual std::shared_ptr<Buffer> chunk(size_t index) const = 0;
    bool is_empty() const;
    void synchronize() const;
};

template<size_t N>
struct ArrayChunk {
    point<N> offset;
    dim<N> shape;
    std::shared_ptr<Buffer> buffer;
};

template<size_t N>
class ArrayImpl {
  public:
    ArrayImpl(dim<N> shape, std::vector<ArrayChunk<N>> chunks);
    ArrayChunk<N> find_chunk(const rect<N>& region) const;

    std::vector<std::shared_ptr<Buffer>> m_buffers;
    dim<N> m_num_chunks;
    dim<N> m_chunk_size;
    dim<N> m_global_size;
};

}  // namespace kmm
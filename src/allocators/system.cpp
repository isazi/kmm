#include "kmm/allocators/system.hpp"

namespace kmm {

bool SystemAllocator::allocate(size_t nbytes, void** addr_out) {
    *addr_out = malloc(nbytes);
    return *addr_out != nullptr;
}

void SystemAllocator::deallocate(void* addr, size_t nbytes) {
    free(addr);
}
}  // namespace kmm
#include "kmm/internals/allocator/system.hpp"

namespace kmm {

bool SystemAllocator::allocate_impl(size_t nbytes, void*& addr_out) {
    addr_out = malloc(nbytes);
    return addr_out != nullptr;
}

void SystemAllocator::deallocate_impl(void* addr_out, size_t nbytes) {
    free(addr_out);
}
}
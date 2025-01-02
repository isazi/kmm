#pragma once

#include <cstddef>
#include <limits>

namespace kmm {

enum struct DeviceMemoryKind {
    /// Use the default memory pool for the device, as returned by `cudaDeviceGetDefaultMemPool`.
    DefaultPool,

    /// Use a newly created private pool for the device, created using `cudaMemPoolCreate`.
    PrivatePool,

    /// No pool. Allocates directly using `cudaMallocAsync` instead of `cudaMallocFromPoolAsync`.
    NoPool,
};

struct WorkerConfig {
    /// Maximum amount of memory that can be allocated on the host, in bytes.
    size_t host_memory_limit = std::numeric_limits<size_t>::max();

    /// If nonzero, use an arena allocator on the host. This will allocate large blocks of the
    /// specified size, which are further split into smaller allocations by the runtime system.
    /// This reduces the number of memory allocation requests to the OS.
    size_t host_memory_block_size = 0;

    /// The type of memory pool to use on the GPU.
    DeviceMemoryKind device_memory_kind = DeviceMemoryKind::DefaultPool;

    /// Maximum amount of memory that can be allocated on each GPU, in bytes. Note that this
    /// specified limit is allowed to exceed the physical memory size of the GPU, in which case
    /// the physical memory is used as the limit instead.
    size_t device_memory_limit = std::numeric_limits<size_t>::max();

    /// If nonzero, use an arena allocator on each device. This will allocate large blocks of the
    /// specified size, from which smaller allocations are subsequently sub-allocated.
    size_t device_memory_block_size = 0;
};

WorkerConfig default_config_from_environment();

}  // namespace kmm
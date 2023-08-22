

# Rationale

We are developing a C++ framework to offload expensive operations for quantitative MRI to GPUs. This framework can be used in C++, but also from other languages, first among them Julia for which a wrapper is already developed.
The idea behind this project is to provide a very light memory manager for the MRI framework, that is general enough to be reused in other contexts as well.

# Design Decisions

## The memory manager owns the memory

Users can request to allocate memory on the GPU, but the GPU memory is managed by kmm. When a user needs the content of allocated memory back, it requests a copy of the content, after this the memory manager can safely deallocate the memory.

**Q**: what if a user already has GPU allocated memory and wants to reuse that?

**Q**: do we return pointers to the user, or implement a virtual pointers system?

## The memory manager does not hold the host buffers

A user can provide a host buffer to copy memory to and from, but the memory manager does not keep track of it, just uses it once. There is no relationship being maintained between host and device buffers.

## Allocation and deallocation should be supported by a memory pool

Having a memory pool can speed-up allocation and deallocation. Using CUDA and CUDA Streams we can use the memory pool available since CUDA 11.2.

**Q**: how to support languages other than CUDA?

# Programming Languages

- C++
- Julia
- Python ?
- CUDA
- HIP ?

# Related Work

Could have we used something else, instead of writing yet another memory manager?


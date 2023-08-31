

# Rationale

We are developing a C++ framework to offload expensive operations for quantitative MRI to GPUs. This framework can be used in C++, but also from other languages, first among them Julia for which a wrapper is already developed.

The idea behind this project (codename KMM) is to provide a very light memory manager for the MRI framework, that is general enough to be reused in other contexts as well.

The memory manager should take care of:
* Allocating the memory *where* necessary
    * The *where* refers to a device such as CPU, GPU, etc.
* Keep track of where the memory is
* Be a proxy for the execution of functions *where* necessary
    * The *where* refers to a device such as CPU, GPU, etc.
* Move the memory to the appropriate device if not there already

# Design Decisions

## The memory manager owns the memory

Users can request to allocate memory on the GPU, but the GPU memory is managed by kmm. When a user needs the content of allocated memory back, it requests a copy of the content, after this the memory manager can safely deallocate the memory.

**Q**: what if a user already has GPU allocated memory and wants to reuse that?
**A**:

**Q**: do we return pointers to the user, or implement a virtual pointers system?
**A**: the solution we adopted is to implement virtual pointers, and keeping track of the physical counterpart (if present) of such pointers

## The memory manager keep track of host buffers only if the user wants

A user can provide a host buffer to copy memory to and from, and only in that case the memory manager keeps track of this buffer, using it if necessary. Because it does not allocate it, it does also not deallocate it.

## Allocation and deallocation should be supported by a memory pool

Having a memory pool can speed-up allocation and deallocation. Using CUDA and CUDA Streams we can use the memory pool available since CUDA 11.2.

**Q**: how to support languages other than CUDA?
**A**: there is experimental support for the same concept in HIP; not sure about other frameworks

## The memory manager owns the memory, not the system

A multi-threaded app may want to create multiple memory managers so that each thread has its own. The memory manager manages its own resources but does not own the GPU, that can then be shared.

# Programming Languages

- C++
- Julia ?
- Python ?
- CUDA
- HIP ?

# Related Work

Could have we used something else, instead of writing yet another memory manager?
This needs some work, but the general idea is that we want something light, not a huge dependency, and want to be able to write code in the language we prefer (current targets CUDA and HIP) and not have another intermediate step in between.

# API

# Implementation Details

## C++

- Using C++ 17
- Not header-only, it needs to be compiled into a shared object to be available in other languages
    - This may change as the focus moves more to C++ than other languages
- The user creates the memory manager, and keeps using it until it is not anymore useful
- The destructor deallocates all memory previously allocated and not released yet

## CUDA

- Each memory manager has a stream associated with, that can be accessed by third-parties that want to schedule kernels on it
- The stream is managed by the memory manager for its all life
- The memory manager does not own the GPU, so device, context and so on need to be stored internally but not exposed

## Julia

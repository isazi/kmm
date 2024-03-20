# KMM

---
[![CPU Build Status](https://github.com/NLeSC-COMPAS/kmm/actions/workflows/cmake-multi-compiler.yml/badge.svg)](https://github.com/NLeSC-COMPAS/kmm/actions/workflows/cmake-multi-compiler.yml)
[![CUDA Build Status](https://github.com/NLeSC-COMPAS/kmm/actions/workflows/cmake-cuda-multi-compiler.yml/badge.svg)](https://github.com/NLeSC-COMPAS/kmm/actions/workflows/cmake-cuda-multi-compiler.yml)
---

KMM is a lightweight C++ middleware for accelerated computing, with native support for CUDA.

Highlights of KMM:
* KMM manages the application memory
  * Allocations of host and device memory is automated
  * Data transfers to and from device are transparently performed when necessary
* No need to rewrite your kernels
  * KMM can schedule and execute native C++ and CUDA functions


## Resources

* [Full documentation](https://nlesc-compas.github.io/kmm)

## License

KMM is made available under the terms of the Apache License version 2.0, see the file LICENSE for details.

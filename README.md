# KMM

KMM is a lightweight C++ middleware for accelerated computing.
The goal of this project is to provide a way to execute code on the main CPU or another accelerator, such as a GPU, without having to perform manual memory operations.
KMM keeps track of where the memory is allocated, and if the content has been modified, in order to perform only the minimum number of memory transfers between host and device.

## Documentation

The documentation for KMM is available [here](https://nlesc-compas.github.io/kmm).
#pragma once

#include <cublas_v2.h>
#include <cuda.h>

#include "system_info.hpp"
#include "task.hpp"
#include "view.hpp"

#include "kmm/utils/checked_math.hpp"
#include "kmm/utils/cuda.hpp"

namespace kmm {

class DeviceContext: public DeviceInfo, public ExecutionContext {
    KMM_NOT_COPYABLE_OR_MOVABLE(DeviceContext);

  public:
    DeviceContext(DeviceInfo info, CudaContextHandle context, CUstream stream);
    ~DeviceContext();

    /**
     * Returns a handle to the CUDA context associated with this device.
     */
    CudaContextHandle context_handle() const {
        return m_context;
    }

    /**
     * Returns a handle to the CUDA context associated with this device.
     */
    CUcontext context() const {
        return m_context;
    }

    /**
     * Returns a handle to the CUDA stream associated with this device.
     */
    CUstream stream() {
        return m_stream;
    }

    /**
     * Shorthand for `stream()`.
     */
    operator CUstream() {
        return m_stream;
    }

    /**
     * Returns a handle to the cuBLAS instance associated with this device.
     */
    cublasHandle_t cublas() const {
        return m_cublas_handle;
    }

    /**
     * Block the current thread until all work submitted onto the stream of this device has
     * finished. Note that the CUDA executor thread will also synchronize the stream automatically
     * after each task, so calling thus function manually is not mandatory.
     */
    void synchronize() const;

    /**
     * Launch the given CUDA kernel onto the stream of this device. The `kernel_function` argument
     * should be a pointer to a `__global__` function.
     */
    template<typename... Args>
    void launch(
        dim3 grid_dim,
        dim3 block_dim,
        unsigned int shared_mem,
        void (*const kernel_function)(Args...),
        Args... args
    ) const {
        // Get void pointer to the arguments.
        void* void_args[sizeof...(Args) + 1] = {static_cast<void*>(&args)..., nullptr};

        // Launch the kernel!
        // NOTE: This must be in the header file since `cudaLaunchKernel` seems to no find the
        // kernel function if it is called from within a C++ file.
        KMM_CUDA_CHECK(cudaLaunchKernel(
            reinterpret_cast<const void*>(kernel_function),
            grid_dim,
            block_dim,
            void_args,
            shared_mem,
            m_stream
        ));
    }

    /**
     * Fill the provided buffer with the copies of the provided value. The fill is performed
     * asynchronously on the stream of this device.
     */
    template<typename T, typename I>
    void fill(T* dest, I num_elements, T value) const {
        fill_bytes(
            dest,
            checked_mul(checked_cast<size_t>(num_elements), sizeof(T)),
            &value,
            sizeof(T)
        );
    }

    /**
     * Fill the provided view with the copies of the provided value. The fill is performed
     * asynchronously on the stream of this device.
     */
    template<typename T, size_t N>
    void fill(cuda_view_mut<T, N> dest, T value) const {
        fill_bytes(dest.data(), dest.size_in_bytes(), &value, sizeof(T));
    }

    /**
     * Copy data from the given source view to the given destination view. The copy is performed
     * asynchronously on the stream of the current device.
     */
    template<typename T, size_t N>
    void copy(cuda_view<T, N> source, cuda_view_mut<T, N> dest) const {
        KMM_ASSERT(source.sizes() == dest.sizes());
        copy_bytes(source.data(), dest.data(), source.size_in_bytes());
    }

    template<typename T, size_t N>
    void copy(cuda_view<T, N> source, view_mut<T, N> dest) const {
        KMM_ASSERT(source.sizes() == dest.sizes());
        copy_bytes(source.data(), dest.data(), source.size_in_bytes());
    }

    template<typename T, size_t N>
    void copy(view<T, N> source, cuda_view_mut<T, N> dest) const {
        KMM_ASSERT(source.sizes() == dest.sizes());
        copy_bytes(source.data(), dest.data(), source.size_in_bytes());
    }

    /**
     * Copy data from the given source view to the given destination view. The copy is performed
     * asynchronously on the stream of the current device.
     */
    template<typename T, typename I>
    void copy(const T* source_ptr, T* dest_ptr, I num_elements) const {
        copy_bytes(
            source_ptr,
            dest_ptr,
            checked_mul(checked_cast<size_t>(num_elements), sizeof(T))
        );
    }

    /**
     * Fill `nbytes` of the buffer starting at `dest_buffer` by repeating the given pattern.
     * The argument `dest_buffer` must be allocated on the device while the `fill_pattern` must
     * be on the host.
     */
    void fill_bytes(
        void* dest_buffer,
        size_t nbytes,
        const void* fill_pattern,
        size_t fill_pattern_size
    ) const;

    /**
     * Copy `nbytes` bytes from the buffer starting at `source_buffer` to the buffer starting at
     * `dest_buffer`. Both buffers must be allocated on the current device.
     */
    void copy_bytes(const void* source_buffer, void* dest_buffer, size_t nbytes) const;

  private:
    CudaContextHandle m_context;
    CUstream m_stream;
    cublasHandle_t m_cublas_handle;
};

}  // namespace kmm
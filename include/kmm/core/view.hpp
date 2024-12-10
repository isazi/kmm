#pragma once

#include "fixed_array.hpp"

namespace kmm {
namespace views {

using default_index_type = int64_t;
using default_stride_type = int32_t;

template<typename I, I... Dims>
struct domain_static_size {
    static constexpr size_t rank = sizeof...(Dims);
    using index_type = I;

    KMM_HOST_DEVICE
    static domain_static_size from_domain(const domain_static_size& domain) noexcept {
        return domain;
    }

    KMM_HOST_DEVICE
    constexpr index_type offset(size_t axis) const noexcept {
        return static_cast<index_type>(0);
    }

    KMM_HOST_DEVICE
    constexpr index_type size(size_t axis) const noexcept {
        index_type sizes[rank] = {Dims...};
        return axis < rank ? sizes[axis] : static_cast<index_type>(1);
    }
};

template<typename I>
struct domain_static_size<I> {
    static constexpr size_t rank = 0;
    using index_type = I;

    template<typename D>
    KMM_HOST_DEVICE static domain_static_size from_domain(const D& domain) noexcept {
        static_assert(D::rank == rank);
        return {};
    }

    KMM_HOST_DEVICE
    constexpr index_type offset(size_t axis) const noexcept {
        return static_cast<index_type>(0);
    }

    KMM_HOST_DEVICE
    constexpr index_type size(size_t axis) const noexcept {
        return static_cast<index_type>(1);
    }
};

template<size_t N, typename I = default_index_type>
struct domain_bounds {
    static constexpr size_t rank = N;
    using index_type = I;

    KMM_HOST_DEVICE
    domain_bounds(fixed_array<index_type, rank> sizes = {}) noexcept : m_sizes(sizes) {}

    KMM_HOST_DEVICE
    static domain_bounds from_domain(const domain_bounds& domain) noexcept {
        return domain;
    }

    template<I... Dims>
    KMM_HOST_DEVICE static domain_bounds from_domain(const domain_static_size<I, Dims...>& domain
    ) noexcept {
        static_assert(sizeof...(Dims) == rank);
        return fixed_array<index_type, rank> {Dims...};
    }

    KMM_HOST_DEVICE
    constexpr index_type offset(size_t axis) const noexcept {
        return static_cast<index_type>(0);
    }

    KMM_HOST_DEVICE
    constexpr index_type size(size_t axis) const noexcept {
        return axis < rank ? m_sizes[axis] : static_cast<index_type>(1);
    }

  private:
    fixed_array<index_type, rank> m_sizes;
};

template<size_t N, typename I = default_index_type>
struct domain_subbounds {
    static constexpr size_t rank = N;
    using index_type = I;

    KMM_HOST_DEVICE
    constexpr domain_subbounds() noexcept {
        for (size_t i = 0; i < N; i++) {
            m_sizes[i] = 0;
            m_offsets[i] = 0;
        }
    }

    KMM_HOST_DEVICE
    constexpr domain_subbounds(fixed_array<index_type, rank> sizes) noexcept : m_sizes(sizes) {
        for (size_t i = 0; i < N; i++) {
            m_offsets[i] = 0;
        }
    }

    KMM_HOST_DEVICE
    constexpr domain_subbounds(
        fixed_array<index_type, rank> offsets,
        fixed_array<index_type, rank> sizes
    ) noexcept :
        m_offsets(offsets),
        m_sizes(sizes) {}

    KMM_HOST_DEVICE
    static domain_subbounds from_domain(const domain_subbounds& domain) noexcept {
        return domain;
    }

    KMM_HOST_DEVICE
    static domain_subbounds from_domain(const domain_bounds<N, I>& domain) noexcept {
        fixed_array<index_type, rank> offsets;
        fixed_array<index_type, rank> sizes;

        for (size_t i = 0; i < rank; i++) {
            offsets[i] = 0;
            sizes[i] = domain.size(i);
        }

        return {offsets, sizes};
    }

    template<I... Dims>
    KMM_HOST_DEVICE static domain_subbounds from_domain(const domain_static_size<I, Dims...>& domain
    ) noexcept {
        return from_domain(domain_bounds<N, I>::from_domain(domain));
    }

    KMM_HOST_DEVICE
    constexpr index_type offset(size_t axis) const noexcept {
        return axis < rank ? m_offsets[axis] : static_cast<index_type>(0);
    }

    KMM_HOST_DEVICE
    constexpr index_type size(size_t axis) const noexcept {
        return axis < rank ? m_sizes[axis] : static_cast<index_type>(1);
    }

  private:
    fixed_array<index_type, rank> m_offsets;
    fixed_array<index_type, rank> m_sizes;
};

template<typename S = default_stride_type, S... Strides>
struct layout_static {
    static constexpr size_t rank = sizeof...(Strides);
    using stride_type = S;

    constexpr layout_static() noexcept = default;

    KMM_HOST_DEVICE static layout_static from_layout(const layout_static& layout) noexcept {
        return {};
    }

    template<typename D>
    KMM_HOST_DEVICE static layout_static from_domain(const D& domain) noexcept {
        static_assert(D::rank == rank);
        return {};
    }

    KMM_HOST_DEVICE
    stride_type stride(size_t axis) const noexcept {
        S strides[rank] = {Strides...};
        return axis < rank ? strides[axis] : static_cast<stride_type>(0);
    }

    template<typename I>
    KMM_HOST_DEVICE ptrdiff_t linearize_index(const fixed_array<I, rank>& ndindex) const noexcept {
        S strides[rank] = {Strides...};
        ptrdiff_t result = 0;

        for (size_t i = 0; i < rank; i++) {
            result += static_cast<ptrdiff_t>(strides[i]) * static_cast<ptrdiff_t>(ndindex[i]);
        }

        return result;
    }
};

template<typename S>
struct layout_static<S> {
    static constexpr size_t rank = 0;
    using stride_type = S;

    template<typename L>
    KMM_HOST_DEVICE static layout_static from_layout(const L& layout) noexcept {
        static_assert(L::rank == 0);
        return {};
    }

    template<typename D>
    KMM_HOST_DEVICE static layout_static from_domain(const D& domain) noexcept {
        static_assert(D::rank == 0);
        return {};
    }

    KMM_HOST_DEVICE
    stride_type stride(size_t axis) const noexcept {
        return static_cast<stride_type>(0);
    }

    template<typename I>
    KMM_HOST_DEVICE ptrdiff_t linearize_index(const fixed_array<I, rank>& ndindex) const noexcept {
        return static_cast<ptrdiff_t>(0);
    }
};

template<typename S = default_stride_type>
using layout_linear = layout_static<S, static_cast<default_stride_type>(1)>;

struct from_strides_t {};

template<size_t N, typename S = default_stride_type>
struct layout_left_to_right {
    static constexpr size_t rank = N;
    using stride_type = S;

    KMM_HOST_DEVICE
    explicit constexpr layout_left_to_right(
        from_strides_t,
        fixed_array<stride_type, rank> strides
    ) noexcept {
        for (size_t i = 1; i < rank; i++) {
            m_strides[i + 1] = strides[i];
        }
    }

    KMM_HOST_DEVICE
    constexpr layout_left_to_right(fixed_array<stride_type, rank> dims) noexcept {
        stride_type stride = 1;

        for (size_t i = 0; i < rank - 1; i++) {
            stride *= static_cast<stride_type>(dims[i]);
            m_strides[i] = stride;
        }
    }

    KMM_HOST_DEVICE
    static layout_left_to_right from_layout(const layout_left_to_right& layout) noexcept {
        return layout;
    }

    template<typename D>
    KMM_HOST_DEVICE static layout_left_to_right from_domain(const D& domain) noexcept {
        fixed_array<stride_type, rank> dims;

        for (size_t i = 0; i < rank; i++) {
            dims[i] = static_cast<stride_type>(domain.size(i));
        }

        return dims;
    }

    KMM_HOST_DEVICE
    stride_type stride(size_t axis) const noexcept {
        if (axis >= 1 && axis < rank) {
            return m_strides[axis - 1];
        } else {
            return static_cast<stride_type>(1);
        }
    }

    template<typename I>
    KMM_HOST_DEVICE ptrdiff_t linearize_index(const fixed_array<I, N>& ndindex) const noexcept {
        ptrdiff_t offset = static_cast<ptrdiff_t>(ndindex[0]);

        for (size_t i = 1; i < rank; i++) {
            offset += static_cast<ptrdiff_t>(m_strides[i - 1]) * static_cast<ptrdiff_t>(ndindex[i]);
        }

        return offset;
    }

  private:
    fixed_array<stride_type, rank - 1> m_strides;
};

template<typename S>
struct layout_left_to_right<0, S>: public layout_static<S> {
    KMM_HOST_DEVICE
    explicit constexpr layout_left_to_right(from_strides_t, fixed_array<S, 0> strides) noexcept {}

    KMM_HOST_DEVICE
    constexpr layout_left_to_right(fixed_array<S, 0> dims = {}) noexcept {}

    KMM_HOST_DEVICE
    constexpr layout_left_to_right(layout_static<S>) noexcept {}
};

template<size_t N, typename S = default_stride_type>
struct layout_right_to_left {
    static constexpr size_t rank = N;
    using stride_type = S;

    KMM_HOST_DEVICE
    explicit constexpr layout_right_to_left(
        from_strides_t,
        fixed_array<stride_type, rank> strides
    ) noexcept {
        for (size_t i = 0; i + 1 < rank; i++) {
            m_strides[i] = strides[i];
        }
    }

    KMM_HOST_DEVICE
    constexpr layout_right_to_left(fixed_array<stride_type, rank> dims) noexcept {
        stride_type stride = 1;

        for (size_t i = 1; i < rank; i++) {
            stride *= dims[N - i];
            m_strides[N - i - 1] = stride;
        }
    }

    KMM_HOST_DEVICE
    static layout_right_to_left from_layout(const layout_right_to_left& layout) noexcept {
        return layout;
    }

    template<typename D>
    KMM_HOST_DEVICE static layout_right_to_left from_domain(const D& domain) noexcept {
        fixed_array<stride_type, rank> dims;

        for (size_t i = 0; i < rank; i++) {
            dims[i] = static_cast<stride_type>(domain.size(i));
        }

        return dims;
    }

    KMM_HOST_DEVICE
    stride_type stride(size_t axis) const noexcept {
        if (axis + 1 < rank) {
            return m_strides[axis];
        } else {
            return static_cast<stride_type>(1);
        }
    }

    template<typename I>
    KMM_HOST_DEVICE ptrdiff_t linearize_index(const fixed_array<I, N>& ndindex) const noexcept {
        ptrdiff_t offset = static_cast<ptrdiff_t>(ndindex[rank - 1]);

        for (size_t i = 0; i + 1 < rank; i++) {
            offset += static_cast<ptrdiff_t>(m_strides[i]) * static_cast<ptrdiff_t>(ndindex[i]);
        }

        return offset;
    }

  private:
    fixed_array<stride_type, rank - 1> m_strides;
};

template<typename S>
struct layout_right_to_left<0, S>: public layout_static<S> {
    KMM_HOST_DEVICE
    explicit constexpr layout_right_to_left(from_strides_t, fixed_array<S, 0> strides) noexcept {}

    KMM_HOST_DEVICE
    constexpr layout_right_to_left(fixed_array<S, 0> dims = {}) noexcept {}

    KMM_HOST_DEVICE
    constexpr layout_right_to_left(layout_static<S>) noexcept {}
};

template<size_t N, typename S = default_stride_type>
struct layout_strided {
    static constexpr size_t rank = N;
    using stride_type = S;

    KMM_HOST_DEVICE
    constexpr layout_strided(fixed_array<stride_type, rank> strides) noexcept :
        m_strides(strides) {}

    template<typename L>
    KMM_HOST_DEVICE static layout_strided from_layout(const L& layout) noexcept {
        fixed_array<stride_type, rank> strides;

        for (size_t i = 0; i < rank; i++) {
            strides[i] = layout.stride(i);
        }

        return strides;
    }

    template<typename D>
    KMM_HOST_DEVICE static layout_strided from_domain(const D& domain) noexcept {
        return from_layout(layout_right_to_left<N, S>::from_domain(domain));
    }

    KMM_HOST_DEVICE
    stride_type stride(size_t axis) const noexcept {
        return m_strides[axis];
    }

    template<typename I>
    KMM_HOST_DEVICE ptrdiff_t linearize_index(const fixed_array<I, N>& ndindex) const noexcept {
        ptrdiff_t result = 0;

        for (size_t i = 0; i < rank; i++) {
            result += static_cast<ptrdiff_t>(m_strides[i]) * static_cast<ptrdiff_t>(ndindex[i]);
        }

        return result;
    }

  private:
    fixed_array<stride_type, rank> m_strides;
};

template<size_t Axis, typename D>
struct drop_domain_axis {
    static_assert(Axis < D::rank);
    using index_type = typename D::index_type;
    using type = domain_subbounds<D::rank - 1, index_type>;

    KMM_HOST_DEVICE
    static type call(const D& domain) noexcept {
        fixed_array<index_type, D::rank - 1> new_offsets;
        fixed_array<index_type, D::rank - 1> new_sizes;
        size_t axis = Axis;

        for (size_t i = 0; i < D::rank - 1; i++) {
            new_offsets[i] = domain.offset(i < axis ? i : i + 1);
            new_sizes[i] = domain.size(i < axis ? i : i + 1);
        }

        return {new_offsets, new_sizes};
    }
};

template<size_t Axis, size_t N, typename I>
struct drop_domain_axis<Axis, domain_bounds<N, I>> {
    static_assert(Axis < N);
    using index_type = I;
    using type = domain_bounds<N - 1, I>;

    KMM_HOST_DEVICE
    static type call(const domain_bounds<N, I>& domain) noexcept {
        fixed_array<index_type, N - 1> new_sizes;
        size_t axis = Axis;

        for (size_t i = 0; i < N - 1; i++) {
            new_sizes[i] = domain.size(i < axis ? i : i + 1);
        }

        return {new_sizes};
    }
};

template<size_t Axis, typename I, I... Dims>
struct drop_domain_axis<Axis, domain_static_size<I, Dims...>> {
    using index_type = I;
    using domain_type = domain_bounds<sizeof...(Dims), I>;
    using type = typename drop_domain_axis<Axis, domain_type>::type;

    KMM_HOST_DEVICE
    static type call(const domain_static_size<I, Dims...>& domain) noexcept {
        return drop_domain_axis<Axis, domain_type>::call(domain_type::from_domain(domain));
    }
};

template<size_t Axis, typename L>
struct drop_layout_axis {
    using stride_type = typename L::stride_type;
    using type = layout_strided<L::rank - 1, stride_type>;

    KMM_HOST_DEVICE
    static type call(const L& layout) noexcept {
        fixed_array<stride_type, L::rank - 1> new_strides;
        size_t axis = Axis;

        for (size_t i = 0; i < L::rank - 1; i++) {
            new_strides[i] = layout.stride(i < axis ? i : i + 1);
        }

        return {new_strides};
    }
};

template<size_t N, typename S>
struct drop_layout_axis<0, layout_right_to_left<N, S>> {
    using stride_type = S;
    using type = layout_right_to_left<N - 1, S>;

    KMM_HOST_DEVICE
    static type call(const layout_right_to_left<N, S>& layout) noexcept {
        fixed_array<stride_type, N - 1> new_strides;

        for (size_t i = 0; i < N - 1; i++) {
            new_strides[i] = layout.stride(i + 1);
        }

        return type {from_strides_t {}, new_strides};
    }
};

template<size_t K, typename S>
struct drop_layout_axis<K, layout_left_to_right<K + 1, S>> {
    using stride_type = S;
    using type = layout_right_to_left<K, S>;

    KMM_HOST_DEVICE
    static type call(const layout_right_to_left<K + 1, S>& layout) noexcept {
        fixed_array<stride_type, K> new_strides;

        for (size_t i = 0; i < K; i++) {
            new_strides[i] = layout.stride(i);
        }

        return type {from_strides_t {}, new_strides};
    }
};

template<size_t N, typename S = default_stride_type>
using layout_default = layout_right_to_left<N, S>;

template<typename D>
using layout_default_for = layout_default<D::rank, typename D::index_type>;

struct accessor_host {
    template<typename T>
    KMM_HOST_DEVICE T& dereference_pointer(T* ptr) const noexcept {
        return *ptr;
    }
};

struct accessor_device {
    template<typename T>
    KMM_HOST_DEVICE T& dereference_pointer(T* ptr) const {
#if __CUDA_ARCH__
        return *ptr;
#else
        throw std::runtime_error("device data cannot be accessed on host");
#endif
    }
};

template<typename A, typename B>
struct convert_pointer;

template<typename T>
struct convert_pointer<T, T> {
    static KMM_HOST_DEVICE T* call(T* p) {
        return p;
    }
};

template<typename T>
struct convert_pointer<T, const T>: convert_pointer<const T, const T> {};

}  // namespace views
template<typename View, typename T, typename D, size_t K = 0, size_t N = D::rank>
struct view_subscript {
    using type = view_subscript;
    using subscript_type = typename view_subscript<View, T, D, K + 1>::type;
    using index_type = typename D::index_type;
    using ndindex_type = fixed_array<index_type, D::rank>;

    KMM_HOST_DEVICE
    static type instantiate(const View* base, ndindex_type index = {}) noexcept {
        return type {base, index};
    }

    KMM_HOST_DEVICE
    view_subscript(const View* base, ndindex_type index) noexcept : base_(base), index_(index) {}

    KMM_HOST_DEVICE
    subscript_type operator[](index_type index) {
        index_[K] = index;
        return view_subscript<View, T, D, K + 1>::instantiate(base_, index_);
    }

  private:
    const View* base_;
    ndindex_type index_;
};

template<typename View, typename T, typename D, size_t N>
struct view_subscript<View, T, D, N, N> {
    using type = T&;
    using index_type = typename D::index_type;
    using ndindex_type = fixed_array<index_type, N>;

    KMM_HOST_DEVICE
    static type instantiate(const View* base, ndindex_type index) {
        return base->access(index);
    }
};

template<typename Derived, typename T, typename D, size_t N = D::rank>
struct basic_view_base {
    using index_type = typename D::index_type;
    using subscript_type = typename view_subscript<Derived, T, D>::subscript_type;

    KMM_HOST_DEVICE
    subscript_type operator[](index_type index) const {
        return view_subscript<Derived, T, D>::instantiate(static_cast<const Derived*>(this))[index];
    }
};

template<typename Derived, typename T, typename D>
struct basic_view_base<Derived, T, D, 0> {
    using reference = T&;

    KMM_HOST_DEVICE
    reference operator*() const {
        return static_cast<const Derived*>(this)->access({});
    }
};

template<typename T, typename D, typename L, typename A = views::accessor_host>
struct basic_view:
    public D,
    public L,
    public A,
    public basic_view_base<basic_view<T, D, L, A>, T, D> {
    static_assert(D::rank == L::rank, "domain type and layout type must have equal rank");

    using self_type = basic_view;
    using value_type = T;
    using domain_type = D;
    using layout_type = L;
    using accessor_type = A;
    using pointer = T*;
    using reference = T&;

    static constexpr size_t rank = D::rank;
    using index_type = typename domain_type::index_type;
    using stride_type = typename layout_type::stride_type;
    using ndindex_type = fixed_array<index_type, rank>;
    using ndstride_type = fixed_array<stride_type, rank>;

    basic_view(const basic_view&) = default;
    basic_view(basic_view&&) noexcept = default;

    basic_view& operator=(const basic_view&) = default;
    basic_view& operator=(basic_view&&) noexcept = default;

    KMM_HOST_DEVICE
    basic_view(
        pointer data,
        domain_type domain,
        layout_type layout,
        accessor_type accessor = {}
    ) noexcept :
        domain_type(domain),
        layout_type(layout),
        accessor_type(accessor) {
        m_data = data - this->layout().linearize_index(offsets());
    }

    KMM_HOST_DEVICE
    basic_view(pointer data = nullptr, domain_type domain = {}) noexcept :
        basic_view(data, domain, layout_type::from_domain(domain)) {}

    template<typename T2, typename D2, typename L2>
    KMM_HOST_DEVICE basic_view(const basic_view<T2, D2, L2, A>& that) noexcept :
        basic_view(
            views::convert_pointer<T2, T>::call(that.data()),
            D::from_domain(that.domain()),
            L::from_layout(that.layout()),
            that.accessor()
        ) {}

    template<typename T2, typename D2, typename L2>
    KMM_HOST_DEVICE basic_view& operator=(const basic_view<T2, D2, L2, A>& that) const noexcept {
        return *this = basic_view(that);
    }

    KMM_HOST_DEVICE
    pointer data() const noexcept {
        return m_data + layout().linearize_index(offsets());
    }

    KMM_HOST_DEVICE
    operator pointer() const noexcept {
        return data();
    }

    KMM_HOST_DEVICE
    const layout_type& layout() const noexcept {
        return *this;
    }

    KMM_HOST_DEVICE
    const domain_type& domain() const noexcept {
        return *this;
    }

    KMM_HOST_DEVICE
    const accessor_type& accessor() const noexcept {
        return *this;
    }

    KMM_HOST_DEVICE
    index_type size(size_t axis) const noexcept {
        return domain().size(axis);
    }

    KMM_HOST_DEVICE
    index_type size() const noexcept {
        index_type volume = 1;
        for (size_t i = 0; i < rank; i++) {
            volume *= domain().size(i);
        }
        return volume;
    }

    KMM_HOST_DEVICE
    size_t size_in_bytes() const noexcept {
        size_t nbytes = sizeof(T);
        for (size_t i = 0; i < rank; i++) {
            nbytes *= static_cast<size_t>(domain().size(i));
        }
        return nbytes;
    }

    KMM_HOST_DEVICE
    stride_type stride(size_t axis = 0) const noexcept {
        return layout().stride(axis);
    }

    KMM_HOST_DEVICE
    index_type offset(size_t axis = 0) const noexcept {
        return domain().offset(axis);
    }

    KMM_HOST_DEVICE
    ndstride_type strides() const noexcept {
        ndstride_type result;
        for (size_t i = 0; i < rank; i++) {
            result[i] = stride(i);
        }
        return result;
    }

    KMM_HOST_DEVICE
    ndindex_type offsets() const noexcept {
        ndindex_type result;
        for (size_t i = 0; i < rank; i++) {
            result[i] = offset(i);
        }
        return result;
    }

    KMM_HOST_DEVICE
    ndindex_type sizes() const noexcept {
        ndindex_type result;
        for (size_t i = 0; i < rank; i++) {
            result[i] = this->size(i);
        }
        return result;
    }

    KMM_HOST_DEVICE
    index_type begin(size_t axis = 0) const noexcept {
        return offset(axis);
    }

    KMM_HOST_DEVICE
    index_type end(size_t axis = 0) const noexcept {
        return begin(axis) + this->size(axis);
    }

    KMM_HOST_DEVICE
    value_type* data_at(ndindex_type ndindex) const noexcept {
        pointer p = m_data;

        for (size_t i = 0; i < rank; i++) {
            p += static_cast<ptrdiff_t>(ndindex[i]) * static_cast<ptrdiff_t>(layout().stride(i));
        }

        return p;
    }

    KMM_HOST_DEVICE
    reference access(ndindex_type ndindex) const noexcept {
        return accessor().dereference_pointer(data_at(ndindex));
    }

    template<typename... Indices>
    KMM_HOST_DEVICE reference operator()(Indices... indices) const noexcept {
        static_assert(sizeof...(Indices) == rank, "invalid number of indices");
        return access(ndindex_type {indices...});
    }

    KMM_HOST_DEVICE
    bool is_empty() const noexcept {
        bool result = false;
        for (size_t i = 0; i < rank; i++) {
            result |= domain().size(i) <= static_cast<index_type>(0);
        }
        return result;
    }

    KMM_HOST_DEVICE
    bool in_bounds(ndindex_type ndindex) const noexcept {
        bool result = true;
        for (size_t i = 0; i < rank; i++) {
            result &= ndindex[i] >= domain().offset(i);
            result &= ndindex[i] - domain().offset(i) < domain().size(i);
        }
        return result;
    }

    KMM_HOST_DEVICE
    bool is_contiguous() const noexcept {
        stride_type curr = 1;
        bool result = true;

        for (size_t i = 0; i < rank; i++) {
            result &= layout().stride(rank - i - 1) == curr;
            curr *= static_cast<stride_type>(domain().size(rank - i - 1));
        }

        return result;
    }

    template<size_t Axis = 0>
    KMM_HOST_DEVICE basic_view<
        T,
        typename views::drop_domain_axis<Axis, domain_type>::type,
        typename views::drop_layout_axis<Axis, layout_type>::type,
        A>
    drop_axis(index_type index) const noexcept {
        static_assert(Axis < rank, "axis out of bounds");
        return {
            data() + layout().stride(Axis) * (index - offset(Axis)),
            views::drop_domain_axis<Axis, domain_type>::call(domain()),
            views::drop_layout_axis<Axis, layout_type>::call(layout()),
        };
    }

    template<size_t Axis = 0>
    KMM_HOST_DEVICE basic_view<
        T,
        typename views::drop_domain_axis<Axis, domain_type>::type,
        typename views::drop_layout_axis<Axis, layout_type>::type,
        A>
    drop_axis() const noexcept {
        static_assert(Axis < rank, "axis out of bounds");
        return this->template drop_axis<Axis>(offset(Axis));
    }

  private:
    pointer m_data;
};

template<
    typename T,
    size_t N = 1,
    typename M = views::layout_default<N>,
    typename A = views::accessor_host>
using view = basic_view<const T, views::domain_bounds<N>, M, A>;

template<
    typename T,
    size_t N = 1,
    typename M = views::layout_default<N>,
    typename A = views::accessor_host>
using view_mut = basic_view<T, views::domain_bounds<N>, M, A>;

template<typename T, size_t N = 1, typename A = views::accessor_host>
using strided_view = view<T, N, views::layout_strided<N>, A>;

template<typename T, size_t N = 1, typename A = views::accessor_host>
using strided_view_mut = view_mut<T, N, views::layout_strided<N>, A>;

template<typename T, size_t N = 1, typename L = views::layout_default<N>>
using gpu_view = view<T, N, L, views::accessor_device>;

template<typename T, size_t N = 1, typename L = views::layout_default<N>>
using gpu_view_mut = view_mut<T, N, L, views::accessor_device>;

template<typename T, size_t N = 1>
using gpu_strided_view = strided_view<T, N, views::accessor_device>;

template<typename T, size_t N = 1>
using gpu_strided_view_mut = strided_view_mut<T, N, views::accessor_device>;

template<
    typename T,
    size_t N = 1,
    typename M = views::layout_default<N>,
    typename A = views::accessor_host>
using subview = basic_view<const T, views::domain_subbounds<N>, M, A>;

template<
    typename T,
    size_t N = 1,
    typename M = views::layout_default<N>,
    typename A = views::accessor_host>
using subview_mut = basic_view<T, views::domain_subbounds<N>, M, A>;

template<typename T, size_t N = 1, typename A = views::accessor_host>
using strided_subview = subview<T, N, views::layout_strided<N>, A>;

template<typename T, size_t N = 1, typename A = views::accessor_host>
using strided_subview_mut = subview_mut<T, N, views::layout_strided<N>, A>;

template<typename T, size_t N = 1, typename L = views::layout_default<N>>
using gpu_subview = subview<T, N, L, views::accessor_device>;

template<typename T, size_t N = 1, typename L = views::layout_default<N>>
using gpu_subview_mut = subview_mut<T, N, L, views::accessor_device>;

template<typename T, size_t N = 1>
using gpu_strided_subview = strided_subview<T, N, views::accessor_device>;

template<typename T, size_t N = 1>
using gpu_strided_subview_mut = strided_subview_mut<T, N, views::accessor_device>;

}  // namespace kmm
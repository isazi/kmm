#include <cstddef>

namespace kmm {

using default_stride_type = ptrdiff_t;
using default_index_type = int;

template<typename T, size_t N>
struct fixed_array {
    constexpr T& operator[](size_t i) {
        return data[i];
    }

    constexpr const T& operator[](size_t i) const {
        return data[i];
    }

    constexpr size_t size() const {
        return N;
    }

    T data[N];
};

template<typename T>
struct fixed_array<T, 0> {
    constexpr T& operator[](size_t i) {
        while (true) {
        }
    }

    constexpr const T& operator[](size_t i) const {
        while (true) {
        }
    }

    constexpr size_t size() const {
        return 0;
    }
};

template<typename T>
struct fixed_array<T, 1> {
    fixed_array(T data = {}) : data(data) {}

    constexpr T& operator[](size_t i) {
        return data;
    }

    constexpr const T& operator[](size_t i) const {
        return data;
    }

    constexpr size_t size() const {
        return 1;
    }

    constexpr operator T() const {
        return data;
    }

    T data;
};

namespace domains {
template<size_t N, typename I = default_index_type>
struct bounds {
    static constexpr size_t rank = N;
    using index_type = I;

    bounds(fixed_array<index_type, rank> sizes = {}) : m_sizes(sizes) {}

    index_type offset(size_t axis) const {
        return 0;
    }

    index_type size(size_t axis) const {
        return axis < rank ? m_sizes[axis] : 1;
    }

  private:
    fixed_array<index_type, rank> m_sizes;
};

template<typename I>
struct bounds<0, I> {
    static constexpr size_t rank = 0;
    using index_type = I;

    bounds(fixed_array<index_type, 0> sizes = {}) {}

    index_type offset(size_t axis) const {
        return 0;
    }

    index_type size(size_t axis) const {
        return 1;
    }
};

template<typename I, I... Sizes>
struct static_bounds {
    static constexpr size_t rank = sizeof...(Sizes);
    using index_type = I;

    constexpr index_type offset(size_t axis) const {
        return 0;
    }

    constexpr index_type size(size_t axis) const {
        fixed_array<index_type, rank> sizes = {Sizes...};
        return axis < rank ? sizes[axis] : 1;
    }
};

template<size_t N, typename I = default_index_type>
struct subbounds {
    static constexpr size_t rank = N;
    using index_type = I;

    subbounds(fixed_array<index_type, rank> offsets, fixed_array<index_type, rank> sizes) :
        m_offsets(offsets),
        m_sizes(sizes) {}

    index_type offset(size_t axis) const {
        return axis < rank ? m_offsets[axis] : 0;
    }

    index_type size(size_t axis) const {
        return axis < rank ? m_sizes[axis] : 1;
    }

  private:
    fixed_array<index_type, rank> m_offsets;
    fixed_array<index_type, rank> m_sizes;
};

template<typename From, typename To, typename = void>
struct convert {};

template<typename D>
struct convert<D, D> {
    static D call(D domain) {
        return domain;
    }
};

template<typename I, I... Sizes>
struct convert<static_bounds<I, Sizes...>, bounds<sizeof...(Sizes), I>> {
    static bounds<sizeof...(Sizes), I> call(static_bounds<I, Sizes...>) {
        return {Sizes...};
    }
};

template<typename D>
struct convert<D, subbounds<D::rank, typename D::index_type>> {
    static constexpr size_t rank = D::rank;
    using index_type = typename D::index_type;

    static subbounds<rank, index_type> call(const D& domain) {
        fixed_array<index_type, rank> new_offsets, new_sizes;
        for (size_t i = 0; i < rank; i++) {
            new_offsets[i] = domain.offset(i);
            new_sizes[i] = domain.size(i);
        }

        return {new_offsets, new_sizes};
    }
};

template<typename D, size_t Axis>
struct remove_axis_impl {};

template<size_t N, typename I, size_t Axis>
struct remove_axis_impl<bounds<N, I>, Axis> {
    static_assert(Axis < N, "Axis cannot exceed rank");
    using type = bounds<N - 1, I>;

    static type call(bounds<N, I> source) {
        fixed_array<I, N - 1> new_sizes;
        for (size_t i = 0; i < N - 1; i++) {
            result[i] = source.size(i + size_t(i >= Axis));
        }
        return new_sizes;
    }
};

template<size_t N, typename I, size_t Axis>
struct remove_axis_impl<subbounds<N, I>, Axis> {
    static_assert(Axis < N, "Axis cannot exceed rank");
    using type = subbounds<N - 1, I>;

    static type call(subbounds<N, I> source) {
        fixed_array<I, N - 1> new_offsets;
        fixed_array<I, N - 1> new_sizes;
        for (size_t i = 0; i < N - 1; i++) {
            new_offsets[i] = source.offset(i + size_t(i >= Axis));
            new_sizes[i] = source.size(i + size_t(i >= Axis));
        }
        return {new_offsets, new_sizes};
    }
};

template<typename D, size_t Axis>
using remove_axis_type = typename remove_axis_impl<D, Axis>::type;
}  // namespace domains

namespace layouts {
template<typename D, typename S = ptrdiff_t>
struct right_to_left: private D {
    static constexpr size_t rank = D::rank;
    using domain_type = D;
    using stride_type = S;
    using index_type = typename domain_type::index_type;

    right_to_left(domain_type domain = {}) : domain_type(domain) {}

    domain_type domain() const {
        return *this;
    }

    stride_type stride(size_t axis) const {
        stride_type stride = 1;

        for (size_t i = rank; i > axis + 1; i--) {
            stride *= stride_type(domain().size(i - 1));
        }

        return stride;
    }

    stride_type linearize_index(fixed_array<index_type, rank> ndindex) const {
        stride_type stride = 1;
        stride_type offset = 0;
        stride_type linear = 0;

        for (size_t i = rank; i > 0; i--) {
            offset += stride_type(domain().offset(i - 1)) * stride;
            linear += stride_type(ndindex[i - 1]) * stride;
            stride *= stride_type(domain().size(i - 1));
        }

        return result;
    }

    index_type required_span() const {
        index_type volume = 1;
        for (size_t i = 0; i < rank; i++) {
            volume *= domain().size(i);
        }
        return volume;
    }
};

template<typename D, typename S = default_stride_type>
struct left_to_right: private D {
    static constexpr size_t rank = D::rank;
    using domain_type = D;
    using stride_type = S;
    using index_type = typename domain_type::index_type;

    left_to_right(domain_type domain = {}) : domain_type(domain) {}

    domain_type domain() const {
        return *this;
    }

    stride_type stride(size_t axis) const {
        stride_type result = 1;
        for (size_t i = 0; i < axis; i++) {
            result *= stride_type {this->size(i)};
        }

        return result;
    }

    stride_type linearize_index(fixed_array<index_type, rank> ndindex) const {
        stride_type stride = 1;
        stride_type offset = 0;
        stride_type linear = 0;

        for (size_t i = 0; i < rank; i++) {
            offset += stride_type(domain().offset(i)) * stride;
            linear += stride_type(ndindex[i]) * stride;
            stride *= stride_type(domain().size(i));
        }

        return linear - offset;
    }

    index_type required_elements() const {
        index_type volume = 1;
        for (size_t i = 0; i < rank; i++) {
            volume *= domain().size(i);
        }
        return volume;
    }
};

template<typename D, typename S = default_stride_type>
struct strided: private D {
    static constexpr size_t rank = D::rank;
    using domain_type = D;
    using stride_type = S;
    using index_type = typename domain_type::index_type;

    strided(domain_type domain = {}, fixed_array<stride_type, rank> strides = {}) :
        domain_type(domain),
        m_strides(strides) {}

    domain_type domain() const {
        return *this;
    }

    stride_type stride(size_t axis) const {
        return axis < rank ? m_strides[axis] : 0;
    }

    stride_type linearize_index(fixed_array<index_type, rank> ndindex) const {
        stride_type offset = 0;
        stride_type linear = 0;

        for (size_t i = 0; i < rank; i--) {
            offset += stride_type(domain().offset(i)) * m_strides[i];
            linear += stride_type(ndindex[i]) * m_strides[i];
        }

        return linear - offset;
    }

    index_type required_elements() const {
        index_type volume = 1;
        bool is_empty = false;

        for (size_t i = 0; i < rank; i++) {
            is_empty = domain().size(i) == 0;
            volume *= (domain().size(i) - 1) * m_strides[i];
        }

        return is_empty ? 0 : volume;
    }

  private:
    fixed_array<stride_type, rank> m_strides;
};

template<typename D, typename S, S... Strides>
struct static_strided: private D {
    static constexpr size_t rank = sizeof...(Strides);
    using domain_type = D;
    using stride_type = S;
    using index_type = typename domain_type::index_type;

    static_strided(domain_type domain = {}) : domain_type(domain) {}

    domain_type domain() const {
        return *this;
    }

    strided<D, S> to_strided() const {
        fixed_array<stride_type, rank> strides = {Strides...};
        return {domain(), strides};
    }

    stride_type stride(size_t axis) const {
        return to_strided().stride(axis);
    }

    stride_type linearize_index(fixed_array<index_type, rank> ndindex) const {
        return to_strided().linearize_index(ndindex);
    }

    index_type required_elements() const {
        return to_strided().required_elements();
    }
};

template<typename From, typename To, typename = void>
struct convert {};

template<typename L>
struct convert<L, L> {
    static L call(const L& layout) {
        return layout;
    }
};

template<typename From, typename D, typename S>
struct convert<From, strided<D, S>> {
    static strided<D> call(const From& from) {
        D new_domain = domains::convert<typename From::domain_type, D>::call(from.domain());

        fixed_array<S, D::rank> new_strides;
        for (size_t i = 0; i < D::rank; i++) {
            new_strides[i] = from.stride(i);
        }

        return {new_domain, new_strides};
    }
};

template<typename D, typename S, typename D2, typename S2>
struct convert<right_to_left<D, S>, right_to_left<D2, S2>> {
    static right_to_left<D2, S2> call(const right_to_left<D, S>& from) {
        return domains::convert<D, D2>::call(from.domain());
    }
};

template<typename D, typename S, typename D2, typename S2>
struct convert<left_to_right<D, S>, left_to_right<D2, S2>> {
    static left_to_right<D2, S2> call(const left_to_right<D, S>& from) {
        return domains::convert<D, D2>::call(from.domain());
    }
};

template<typename L, size_t Axis>
struct remove_axis_impl {
    using new_domain_type = domains::remove_axis_type<typename L::domain_type, Axis>;
    using stride_type = typename L::stride_type;
    using type = strided<new_domain_type, stride_type>;

    static type call(const L& from) {
        auto new_domain = domains::remove_axis_impl<typename L::domain_type, Axis>::call(from);

        fixed_array<stride_type, L::rank - 1> new_strides;
        for (size_t i = 0; i < L::rank - 1; i++) {
            new_strides[i] = from.stride(i + size_t(i >= Axis));
        }

        return {new_domain, new_strides};
    }
};

template<typename D, typename S>
struct remove_axis_impl<right_to_left<D, S>, 0> {
    using type = right_to_left<domains::remove_axis_type<D, 0>, S>;

    static type call(const right_to_left<D, S>& from) {
        return {domains::remove_axis_impl<D, 0>::call(from)};
    }
};

template<typename L, size_t Axis>
using remove_axis_type = typename remove_axis_impl<L, Axis>::type;

}  // namespace layouts

namespace accessors {
struct host {
    template<typename T>
    T* offset(T* ptr, ptrdiff_t offset) {
        return ptr + offset;
    }

    template<typename T>
    T& access(T* ptr) {
        return *ptr;
    }
};

struct cuda_device {
    template<typename T>
    T* offset(T* ptr, ptrdiff_t offset) {
        return ptr + offset;
    }

    template<typename T>
    T& access(T* ptr) {
        return *ptr;
    }
};

template<typename From, typename To>
struct convert {};

template<typename T>
struct convert<T, T> {
    static T* call(T* input) {
        return input;
    }
};

template<typename T>
struct convert<T, const T> {
    static const T* call(T* input) {
        return input;
    }
};
}  // namespace accessors

template<typename V, size_t I = 0, size_t N = V::rank>
struct view_subscript {
    using type = view_subscript;
    using subscript_type = typename view_subscript<V, I + 1>::type;
    using index_type = typename V::index_type;
    using ndindex_type = fixed_array<index_type, N>;

    static type instantiate(const V* base, ndindex_type index = {}) {
        return type {base, index};
    }

    view_subscript(const V* base, ndindex_type index) : base_(base), index_(index) {}

    subscript_type operator[](index_type index) {
        index_[I] = index;
        return view_subscript<V, I + 1>::instantiate(base_, index_);
    }

  private:
    const V* base_;
    ndindex_type index_;
};

template<typename V, size_t N>
struct view_subscript<V, N, N> {
    using type = typename V::reference;
    using index_type = typename V::index_type;
    using ndindex_type = fixed_array<index_type, N>;

    static type instantiate(V* base, ndindex_type index) {
        return base->access(index);
    }
};

template<typename Derived, size_t N>
struct basic_view_base {
    using index_type = typename Derived::index_type;
    using subscript_type = typename view_subscript<Derived>::subscript_type;

    subscript_type operator[](index_type index) const {
        return view_subscript<Derived> {static_cast<const Derived*>(this)}[index];
    }
};

template<typename Derived>
struct basic_view_base<Derived, 0> {
    using reference = typename Derived::reference;

    reference operator*() const {
        return static_cast<const Derived*>(this)->access({});
    }
};

template<typename T, typename L, typename A>
struct basic_view: private L, private A, public basic_view_base<basic_view<T, L, A>, L::rank> {
    using self_type = basic_view;
    using value_type = T;
    using layout_type = L;
    using accessor_type = A;
    using pointer = T*;
    using reference = T&;

    using domain_type = typename layout_type::domain_type;
    using stride_type = typename layout_type::stride_type;
    using index_type = typename domain_type::index_type;
    static constexpr size_t rank = domain_type::rank;
    using ndindex_type = fixed_array<index_type, rank>;
    using ndstride_type = fixed_array<stride_type, rank>;

    basic_view(const basic_view&) = default;

    explicit basic_view(
        pointer data = nullptr,
        layout_type layout = {},
        accessor_type accessor = {}) :
        m_data(data),
        layout_type(layout),
        accessor_type(accessor) {}

    basic_view(pointer data, domain_type domain, accessor_type accessor = {}) :
        basic_view(data, layout_type(domain), accessor) {}

    template<typename T2, typename L2>
    basic_view(const basic_view<T2, L2, A>& that) :
        m_data(accessors::convert<T2, T>::call(that.data())),
        layout_type(layouts::convert<L2, L>::call(that.layout())),
        accessor_type(that.accessor()) {}

    pointer data() const {
        return m_data;
    }

    operator pointer() const {
        return data();
    }

    const layout_type& layout() const {
        return *this;
    }

    const accessor_type& accessor() const {
        return *this;
    }

    const domain_type& domain() const {
        return layout().domain();
    }

    stride_type stride(size_t axis = 0) const {
        return layout().stride(axis);
    }

    index_type offset(size_t axis = 0) const {
        return domain().offset(axis);
    }

    index_type size(size_t axis) const {
        return domain().size(axis);
    }

    ndstride_type strides() const {
        ndstride_type result;
        for (size_t i = 0; i < rank; i++) {
            result[i] = stride(i);
        }
        return result;
    }

    ndindex_type offsets() const {
        ndindex_type result;
        for (size_t i = 0; i < rank; i++) {
            result[i] = offset(i);
        }
        return result;
    }

    ndindex_type sizes() const {
        ndindex_type result;
        for (size_t i = 0; i < rank; i++) {
            result[i] = size(i);
        }
        return result;
    }

    index_type size() const {
        index_type total = 0;
        for (size_t i = 0; i < rank; i++) {
            total *= size(i);
        }
        return total;
    }

    bool is_empty() const {
        bool result = false;
        for (size_t i = 0; i < rank; i++) {
            result |= size(i) == 0;
        }
        return result;
    }

    bool in_bounds(ndindex_type point) const {
        bool result = true;
        for (size_t i = 0; i < rank; i++) {
            result &= point[i] >= begin(i);
            result &= point[i] - begin(i) < size(i);
        }
        return result;
    }

    index_type begin(index_type axis) const {
        return offset(axis);
    }

    index_type end(index_type axis) const {
        return begin(axis) + size(axis);
    }

    stride_type linearize_index(ndindex_type point) const {
        return layout().linearize_index(point);
    }

    value_type* data_at(ndindex_type point) const {
        return accessor().offset(m_data, linearize_index(point));
    }

    reference access(ndindex_type point) const {
        return accessor().access(data_at(point));
    }

    template<size_t Axis>
    basic_view<value_type, layouts::remove_axis_type<L, Axis>, accessor_type> drop_axis(
        index_type index) {
        stride_type offset = stride_type(index - begin(Axis)) * stride(Axis);
        return {
            accessor().offset(m_data, offset),
            layouts::remove_axis_impl<L, Axis>::call(layout())};
    }

    template<size_t Axis>
    basic_view<value_type, layouts::remove_axis_type<L, Axis>, accessor_type> drop_axis() {
        return {m_data, layouts::remove_axis_impl<L, Axis>::call(layout())};
    }

    template<typename... Indices>
    reference operator()(Indices... indices) const {
        static_assert(sizeof...(Indices) == rank, "invalid number of indices");
        return access(ndindex_type {indices...});
    }

  private:
    pointer m_data;
};

template<
    typename T,
    size_t N = 1,
    template<class D> typename L = layouts::right_to_left,
    typename A = accessors::host>
using view = basic_view<const T, L<domains::bounds<N>>, A>;

template<
    typename T,
    size_t N = 1,
    template<class D> typename L = layouts::right_to_left,
    typename A = accessors::host>
using view_mut = basic_view<T, L<domains::bounds<N>>, A>;

template<typename T, size_t N = 1, typename A = accessors::host>
using strided_view = view<T, N, layouts::strided, A>;

template<typename T, size_t N = 1, typename A = accessors::host>
using strided_view_mut = view_mut<T, N, layouts::strided, A>;

template<typename T, size_t N = 1, template<class D> typename L = layouts::right_to_left>
using cuda_view = view<T, N, L, accessors::cuda_device>;

template<typename T, size_t N = 1, template<class D> typename L = layouts::right_to_left>
using cuda_view_mut = view_mut<T, N, L, accessors::cuda_device>;

template<typename T, size_t N = 1>
using cuda_strided_view = strided_view<T, N, accessors::cuda_device>;

template<typename T, size_t N = 1>
using cuda_strided_view_mut = strided_view_mut<T, N, accessors::cuda_device>;

}  // namespace kmm
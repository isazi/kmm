#pragma once

#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>

namespace kmm {

[[noreturn]] __attribute__((noinline)) inline void throw_small_vector_out_of_capacity() {
    throw std::overflow_error("small_vector exceeds capacity");
}

template<typename T, size_t InlineSize>
struct small_vector {
    using capacity_type = uint32_t;

    small_vector() = default;

    small_vector(const small_vector& that) {
        insert_all(that.begin(), that.end());
    }

    template<size_t N>
    small_vector(const small_vector<T, N>& that) {
        insert_all(that.begin(), that.end());
    }

    small_vector(small_vector&& that) noexcept {
        *this = std::move(that);
    }

    small_vector(std::initializer_list<T> items) {
        insert_all(items.begin(), items.end());
    }

    template<typename U, size_t K>
    small_vector& operator=(const small_vector<U, K>& that) {
        if (this != &that) {
            clear();
            insert_all(that);
        }

        return *this;
    }

    small_vector& operator=(const small_vector& that) {
        if (this != &that) {
            clear();
            insert_all(that);
        }

        return *this;
    }

    small_vector& operator=(small_vector&& that) noexcept {
        std::swap(this->m_inline_data, that.m_inline_data);
        std::swap(this->m_size, that.m_size);
        std::swap(this->m_capacity, that.m_capacity);
        std::swap(this->m_data, that.m_data);

        if (!this->is_heap_allocated()) {
            this->m_data = this->m_inline_data;
        }

        if (!that.is_heap_allocated()) {
            that.m_data = that.m_inline_data;
        }

        return *this;
    }

    ~small_vector() {
        if (is_heap_allocated()) {
            delete[] m_data;
        }
    }

    size_t capacity() const noexcept {
        return m_capacity;
    }

    size_t size() const noexcept {
        return m_size;
    }

    bool is_empty() const noexcept {
        return m_size == 0;
    }

    void clear() noexcept {
        m_size = 0;
    }

    bool is_heap_allocated() const noexcept {
        return m_capacity > InlineSize;
    }

    T* data() noexcept {
        return m_data;
    }

    const T* data() const noexcept {
        return m_data;
    }

    bool try_grow_capacity(size_t k = 1) {
        capacity_type new_capacity = m_capacity;
        do {
            if (new_capacity > std::numeric_limits<capacity_type>::max() - new_capacity) {
                return false;
            }

            new_capacity += new_capacity;
        } while (new_capacity - m_size < k);

        if (new_capacity < 16) {
            new_capacity = 16;
        }

        auto new_data = std::unique_ptr<T[]>(new (std::nothrow_t {}) T[new_capacity]);

        if (new_data == nullptr) {
            return false;
        }

        for (size_t i = 0; i < m_size; i++) {
            new_data[i] = std::move(m_data[i]);
        }

        if (is_heap_allocated()) {
            delete[] m_data;
        }

        m_capacity = new_capacity;
        m_data = new_data.release();
        return true;
    }

    bool try_push_back(T item) {
        if (m_capacity <= m_size && !try_grow_capacity(1)) {
            return false;
        }

        m_data[m_size] = std::move(item);
        m_size++;
        return true;
    }

    void push_back(T item) {
        if (!try_push_back(std::move(item))) {
            throw_small_vector_out_of_capacity();
        }
    }

    template<typename It>
    void insert_all(It begin, It end) {
        size_t n = static_cast<size_t>(end - begin);

        if (m_capacity - m_size < n) {
            if (!try_grow_capacity(n)) {
                throw_small_vector_out_of_capacity();
            }
        }

        for (size_t i = 0; i < n; i++) {
            m_data[m_size + i] = begin[i];
        }

        // This is safe since `n <= m_capacity - m_size`
        m_size += static_cast<capacity_type>(n);
    }

    void insert_all(small_vector&& that) {
        if (is_empty()) {
            *this = std::move(that);
            return;
        }

        insert_all(that.begin(), that.end());
    }

    template<typename U, size_t K>
    void insert_all(const small_vector<U, K>& that) {
        insert_all(that.begin(), that.end());
    }

    void resize(size_t n) {
        if (m_capacity < n) {
            if (!try_grow_capacity(n - m_size)) {
                throw_small_vector_out_of_capacity();
            }
        }

        // Safe since `n <= m_capacity`
        m_size = static_cast<capacity_type>(n);
    }

    void truncate(size_t new_size) {
        if (new_size > m_size) {
            return;
        }

        // Safe since `new_size < m_size`
        m_size = static_cast<capacity_type>(new_size);
    }

    T& operator[](size_t i) noexcept {
        return *(data() + i);
    }

    T* begin() noexcept {
        return data();
    }

    T* end() noexcept {
        return data() + size();
    }

    const T& operator[](size_t i) const noexcept {
        return *(data() + i);
    }

    const T* begin() const noexcept {
        return data();
    }

    const T* end() const noexcept {
        return data() + size();
    }

    template<typename F>
    const T* find_if(F pred) const {
        for (const auto& v : *this) {
            if (pred(v)) {
                return &v;
            }
        }

        return end();
    }

    template<typename R>
    bool contains(R&& item) const {
        bool found = false;

        for (const auto& v : *this) {
            if (v == item) {
                found = true;
            }
        }

        return found;
    }

  private:
    capacity_type m_size = 0;
    capacity_type m_capacity = InlineSize;
    T m_inline_data[InlineSize];
    T* m_data = m_inline_data;
};

using small_buffer = small_vector<uint8_t, sizeof(uint64_t)>;

}  // namespace kmm
#pragma once

#include <cstddef>
#include <memory>

namespace kmm {

template<typename T, size_t InlineSize>
struct small_vector {
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

    size_t capacity() const {
        return m_capacity;
    }

    size_t size() const {
        return m_size;
    }

    bool is_empty() const {
        return m_size == 0;
    }

    void clear() {
        m_size = 0;
    }

    bool is_heap_allocated() const {
        return m_capacity > InlineSize;
    }

    T* data() {
        return m_data;
    }

    const T* data() const {
        return m_data;
    }

    void grow_capacity(size_t k = 1) {
        uint32_t new_capacity = m_capacity;
        do {
            if (new_capacity >= uint32_t(0x100000000LL / 2)) {
                throw std::overflow_error("small_vector exceeds capacity");
            }

            new_capacity *= 2;
        } while (new_capacity - m_size < k);

        if (new_capacity < 16) {
            new_capacity = 16;
        }

        auto new_data = std::make_unique<T[]>(new_capacity);

        for (size_t i = 0; i < m_size; i++) {
            new_data[i] = std::move(m_data[i]);
        }

        if (is_heap_allocated()) {
            delete[] m_data;
        }

        m_capacity = new_capacity;
        m_data = new_data.release();
    }

    void push_back(T item) {
        if (m_capacity <= m_size) {
            grow_capacity();
        }

        m_data[m_size] = std::move(item);
        m_size++;
    }

    template<typename U>
    void insert_all(const U* begin, const U* end) {
        size_t n = end - begin;

        if (m_capacity - m_size < n) {
            grow_capacity(n);
        }

        for (size_t i = 0; i < n; i++) {
            m_data[m_size + i] = begin[i];
        }

        m_size += n;
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
            grow_capacity(n - m_size);
        }

        m_size = n;
    }

    T& operator[](size_t i) {
        return *(data() + i);
    }

    T* begin() {
        return data();
    }

    T* end() {
        return data() + size();
    }

    const T& operator[](size_t i) const {
        return *(data() + i);
    }

    const T* begin() const {
        return data();
    }

    const T* end() const {
        return data() + size();
    }

    bool remove(const T& item) {
        T* p = data();

        size_t i = 0;
        bool found = false;

        for (; i < size(); i++) {
            if (p[i] == item) {
                found = true;
                break;
            }
        }

        if (!found) {
            return false;
        }

        for (; i < size(); i++) {
            p[i - 1] = p[i];
        }

        m_size--;
        return true;
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
    uint32_t m_size = 0;
    uint32_t m_capacity = InlineSize;
    T m_inline_data[InlineSize];
    T* m_data = m_inline_data;
};

}  // namespace kmm
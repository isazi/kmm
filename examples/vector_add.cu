#include "kmm/api/runtime.hpp"
#include "kmm/api/array.hpp"

void kernel_function(
    kmm::rect<2> subregion,
    int n, int m, int k,
    kmm::subview<float, 2> a,
    kmm::subview<float, 2> b,
    kmm::subview_mut<float, 2> c
) {
    for (int i = subregion.begin(0); i < subregion.end(0); i++) {
        for (int j = subregion.begin(1); j < subregion.end(1); j++) {
            for (int l = 0; l < k; l++) {
                c[i][j] += a[i][l] * b[l][j];
            }
        }
    }
}

void example(kmm::Runtime rt) {
    using namespace kmm::placeholders;

    int n = 150;
    int m = 50;
    int k = 10;

    kmm::Array<float, 2> a = {n, k};
    kmm::Array<float, 2> b = {k, m};
    kmm::Array<float, 2> c = {n, m};

    rt.submit(
        kmm::Host(),
        kmm::ChunkPartition<2> {
            {n, m},
            {10, 10}
        },
        kernel_function,
        n,
        m,
        k,
        read(a, slice(_x, _)),
        read(b, slice(_, _y)),
        write(c, slice(_x, _y))
    );
}

int main() {
    auto rt = kmm::make_runtime();
    example(rt);
}
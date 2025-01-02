#include "kmm/core/reduction.hpp"

namespace kmm {

std::ostream& operator<<(std::ostream& f, Reduction p) {
    switch (p) {
        case Reduction::Sum:
            return f << "Sum";
        case Reduction::Product:
            return f << "Product";
        case Reduction::Min:
            return f << "Min";
        case Reduction::Max:
            return f << "Max";
        case Reduction::BitAnd:
            return f << "BitAnd";
        case Reduction::BitOr:
            return f << "BitOr";
        default:
            return f << "(unknown operation)";
    }
}

std::ostream& operator<<(std::ostream& f, ReductionOutput p) {
    return f << "Reduction(" << p.operation << ", " << p.data_type << ")";
}

}  // namespace kmm
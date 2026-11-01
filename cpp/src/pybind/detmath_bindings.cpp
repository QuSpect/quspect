#include "detmath_bindings.hpp"

#include <complex>
#include <cstring>
#include <stdexcept>
#include <string>

#include <pybind11/numpy.h>

#include "quspect_native/detmath/complex.hpp"
#include "quspect_native/detmath/linalg.hpp"
#include "quspect_native/detmath/scalar.hpp"

namespace py = pybind11;

namespace quspect::native::pybind {

namespace {

namespace dm = quspect::native::detmath;

using RealArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using ComplexArray = py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast>;

// 对数组逐元素求 detmath 标量函数（二元函数 atan2 / hypot 的第二个参数为 y）。
py::array_t<double> Scalar(const std::string& name, const RealArray& x, const RealArray& y) {
    const auto n = static_cast<std::size_t>(x.size());
    const bool binary = name == "atan2" || name == "hypot" || name == "pow";
    if (binary && static_cast<std::size_t>(y.size()) != n) {
        throw std::invalid_argument("operand sizes differ");
    }
    double (*unary)(double) = nullptr;
    if (name == "sin") {
        unary = dm::Sin;
    } else if (name == "cos") {
        unary = dm::Cos;
    } else if (name == "tan") {
        unary = dm::Tan;
    } else if (name == "asin") {
        unary = dm::Asin;
    } else if (name == "acos") {
        unary = dm::Acos;
    } else if (name == "atan") {
        unary = dm::Atan;
    } else if (name == "exp") {
        unary = dm::Exp;
    } else if (name == "log") {
        unary = dm::Log;
    } else if (!binary) {
        throw std::invalid_argument("unknown detmath function: " + name);
    }
    const double* px = x.data();
    const double* py_ = binary ? y.data() : nullptr;
    py::array_t<double> out(static_cast<py::ssize_t>(n));
    double* po = out.mutable_data();
    for (std::size_t i = 0; i < n; ++i) {
        if (name == "atan2") {
            po[i] = dm::Atan2(px[i], py_[i]);
        } else if (name == "hypot") {
            po[i] = dm::Hypot(px[i], py_[i]);
        } else if (name == "pow") {
            po[i] = dm::Pow(px[i], py_[i]);
        } else {
            po[i] = unary(px[i]);
        }
    }
    return out;
}

const dm::Cx* Data(const ComplexArray& a, int* n) {
    if (a.ndim() != 2 || a.shape(0) != a.shape(1) || a.shape(0) < 1 || a.shape(0) > dm::kMaxDim) {
        throw std::invalid_argument("expected a square matrix of size 1..8");
    }
    *n = static_cast<int>(a.shape(0));
    return reinterpret_cast<const dm::Cx*>(a.data());
}

py::array_t<std::complex<double>> NewMatrix(int n, const dm::Cx* data) {
    py::array_t<std::complex<double>> out({n, n});
    std::memcpy(out.mutable_data(), data, sizeof(dm::Cx) * static_cast<std::size_t>(n * n));
    return out;
}

py::tuple Svd(const ComplexArray& a) {
    int n = 0;
    const dm::Cx* pa = Data(a, &n);
    dm::Cx u[dm::kMaxDim * dm::kMaxDim];
    dm::Cx vh[dm::kMaxDim * dm::kMaxDim];
    double s[dm::kMaxDim];
    if (!dm::Svd(n, pa, u, s, vh)) {
        throw std::runtime_error("detmath SVD did not converge");
    }
    py::array_t<double> sv(n);
    std::memcpy(sv.mutable_data(), s, sizeof(double) * static_cast<std::size_t>(n));
    return py::make_tuple(NewMatrix(n, u), sv, NewMatrix(n, vh));
}

py::tuple Eigh(const RealArray& a) {
    if (a.ndim() != 2 || a.shape(0) != a.shape(1) || a.shape(0) < 1 || a.shape(0) > dm::kMaxDim) {
        throw std::invalid_argument("expected a square matrix of size 1..8");
    }
    const int n = static_cast<int>(a.shape(0));
    py::array_t<double> w(n);
    py::array_t<double> v({n, n});
    if (!dm::EighSym(n, a.data(), w.mutable_data(), v.mutable_data())) {
        throw std::runtime_error("detmath eigh did not converge");
    }
    return py::make_tuple(w, v);
}

py::tuple Qr(const ComplexArray& a) {
    int n = 0;
    const dm::Cx* pa = Data(a, &n);
    dm::Cx q[dm::kMaxDim * dm::kMaxDim];
    dm::Cx r[dm::kMaxDim * dm::kMaxDim];
    dm::Qr(n, pa, q, r);
    return py::make_tuple(NewMatrix(n, q), NewMatrix(n, r));
}

py::array_t<std::complex<double>> MatMul(const ComplexArray& a, const ComplexArray& b) {
    int n = 0;
    int nb = 0;
    const dm::Cx* pa = Data(a, &n);
    const dm::Cx* pb = Data(b, &nb);
    if (n != nb) {
        throw std::invalid_argument("matrix sizes differ");
    }
    dm::Cx out[dm::kMaxDim * dm::kMaxDim];
    dm::MatMul(n, pa, pb, out);
    return NewMatrix(n, out);
}

std::complex<double> Det(const ComplexArray& a) {
    int n = 0;
    const dm::Cx* pa = Data(a, &n);
    const dm::Cx d = dm::Det(n, pa);
    return {d.re, d.im};
}

}  // namespace

void BindDetmath(py::module_& m) {
    m.def("_detmath_scalar", &Scalar, py::arg("name"), py::arg("x"), py::arg("y") = RealArray(0),
          "Correctly rounded scalar function of the deterministic numerics layer, elementwise.");
    m.def("_detmath_svd", &Svd, py::arg("a"), "Deterministic one-sided Jacobi SVD of a small complex matrix.");
    m.def("_detmath_eigh", &Eigh, py::arg("a"), "Deterministic Jacobi eigendecomposition of a real symmetric matrix.");
    m.def("_detmath_qr", &Qr, py::arg("a"),
          "Deterministic Householder QR (R with a non-negative real diagonal) of a small complex matrix.");
    m.def("_detmath_matmul", &MatMul, py::arg("a"), py::arg("b"), "Deterministic product of two small complex matrices.");
    m.def("_detmath_det", &Det, py::arg("a"), "Deterministic determinant (partial-pivoting LU).");
}

}  // namespace quspect::native::pybind

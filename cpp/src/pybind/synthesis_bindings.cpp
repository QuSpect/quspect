#include "synthesis_bindings.hpp"

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "quspect_native/detmath/linalg.hpp"
#include "quspect_native/synthesis/gate_matrices.hpp"
#include "quspect_native/synthesis/kak_synthesis.hpp"
#include "quspect_native/synthesis/one_qubit_synthesis.hpp"
#include "quspect_native/synthesis/three_qubit.hpp"

namespace py = pybind11;

namespace quspect::native::pybind {

namespace {

using namespace quspect::native::synthesis;

using ComplexArray = py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast>;
// 不做转换：只接受本来就是 C 连续 complex128 的数组。
using StrictArray = py::array_t<std::complex<double>, py::array::c_style>;
using RealArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

static_assert(sizeof(Cx) == sizeof(std::complex<double>), "Cx must match complex128");

template <std::size_t N>
std::array<Cx, N> ToMatrix(const ComplexArray& arr) {
    if (static_cast<std::size_t>(arr.size()) != N) {
        throw std::invalid_argument("unexpected matrix size");
    }
    std::array<Cx, N> out;
    std::memcpy(out.data(), arr.data(), N * sizeof(Cx));
    return out;
}

template <std::size_t N>
py::array_t<std::complex<double>> FromMatrix(const std::array<Cx, N>& m) {
    const auto n = static_cast<py::ssize_t>(N == 4 ? 2 : 4);
    py::array_t<std::complex<double>> out({n, n});
    std::memcpy(out.mutable_data(), m.data(), N * sizeof(Cx));
    return out;
}

// 合成结果 → Python 的 (name, qubits, params) 元组列表（与 two_qubit.SynthOp 相同）。
class OpConverter {
 public:
    OpConverter()
        : rz_("rz"),
          h_("h"),
          x_("x"),
          sx_("sx"),
          q0_(py::make_tuple(0)),
          q1_(py::make_tuple(1)),
          q01_(py::make_tuple(0, 1)),
          q10_(py::make_tuple(1, 0)),
          empty_(py::tuple()) {}

    py::list Convert(const std::vector<Op>& ops, const py::str& entangler) const {
        py::list out(ops.size());
        for (std::size_t i = 0; i < ops.size(); ++i) {
            const Op& op = ops[i];
            const py::tuple& q = op.q0 == 0 ? q0_ : q1_;
            py::tuple item;
            switch (op.kind) {
                case OpKind::kEntangler:
                    item = py::make_tuple(entangler, op.q0 == 0 ? q01_ : q10_, empty_);
                    break;
                case OpKind::kRz:
                    item = py::make_tuple(rz_, q, py::make_tuple(op.param));
                    break;
                case OpKind::kH:
                    item = py::make_tuple(h_, q, empty_);
                    break;
                case OpKind::kX:
                    item = py::make_tuple(x_, q, empty_);
                    break;
                case OpKind::kSX:
                    item = py::make_tuple(sx_, q, empty_);
                    break;
            }
            out[i] = std::move(item);
        }
        return out;
    }

 private:
    py::str rz_, h_, x_, sx_;
    py::tuple q0_, q1_, q01_, q10_, empty_;
};

// 两比特 KAK 合成、单比特合成与块合并的辅助函数（确定性数值层，全平台逐位一致）。
class PyKakSynthesizer {
 public:
    PyKakSynthesizer() = default;

    int AddEntangler(const std::string& name, const ComplexArray& matrix, const ComplexArray& b0,
                     const ComplexArray& b1, const ComplexArray& a0, const ComplexArray& a1, bool symmetric) {
        EntanglerSpec spec;
        spec.name = name;
        spec.matrix = ToMatrix<16>(matrix);
        spec.before = {ToMatrix<4>(b0), ToMatrix<4>(b1)};
        spec.after = {ToMatrix<4>(a0), ToMatrix<4>(a1)};
        spec.symmetric = symmetric;
        names_.emplace_back(name);
        return impl_.AddEntangler(std::move(spec));
    }

    // _synthesize_two_qubit_ops：(ops, uses_flag)。不是酉矩阵时抛出 ValueError，合成失败时抛出 RuntimeError
    // （与 Python 参考实现相同的异常与消息）。
    py::tuple Synthesize(const ComplexArray& u, int entangler, int euler, double atol, double verify_atol,
                         bool swap_face_template, bool mirrored) const {
        const SynthOptions opt = Options(entangler, euler, atol, verify_atol, swap_face_template, mirrored);
        const M4 m = ToMatrix<16>(u);
        impl_.CheckUnitary(m);
        bool uses_flag = false;
        const std::vector<Op> ops = impl_.SynthesizeExact(m, opt, &uses_flag);
        return py::make_tuple(conv_.Convert(ops, Name(entangler)), uses_flag);
    }

    // synthesize_two_qubit_approx 的前半段：(0, ops | None, uses_flag) = 没有可用的近似（want_exact
    // 时附上精确合成）；(1, None, False) = 可能有近似（由调用方处理）。
    py::tuple Approximate(const ComplexArray& u, double max_infidelity, double atol, double verify_atol,
                          int max_entanglers, int entangler, int euler, bool swap_face_template, bool mirrored,
                          bool want_exact) const {
        const SynthOptions opt = Options(entangler, euler, atol, verify_atol, swap_face_template, mirrored);
        const M4 m = ToMatrix<16>(u);
        impl_.CheckUnitary(m);
        const Kak kak = impl_.Kak4(m);
        if (!impl_.ApproximationImpossible(kak, max_infidelity, atol, max_entanglers)) {
            return py::make_tuple(1, py::none(), false);
        }
        if (!want_exact) {
            return py::make_tuple(0, py::none(), false);
        }
        bool uses_flag = false;
        const std::vector<Op> ops = impl_.SynthesizeExact(m, kak, opt, &uses_flag);
        return py::make_tuple(0, conv_.Convert(ops, Name(entangler)), uses_flag);
    }

    // _synthesize_swap_face_block：ops；False = 不在 SWAP 面上或未通过验证（Python 返回 None）。
    py::object SwapFace(const ComplexArray& u, int entangler, int euler, double verify_atol, bool mirrored) const {
        const SynthOptions opt = Options(entangler, euler, 1e-9, verify_atol, false, mirrored);
        const M4 m = ToMatrix<16>(u);
        impl_.CheckUnitary(m);
        const auto ops = impl_.SynthesizeSwapFace(m, opt);
        if (!ops) {
            return py::bool_(false);
        }
        return conv_.Convert(*ops, Name(entangler));
    }

    // _count_from_coords(_kak4(u)[2], atol)。
    int NumCz(const ComplexArray& u, double atol) const {
        const M4 m = ToMatrix<16>(u);
        impl_.CheckUnitary(m);
        return KakSynthesizer::CountFromCoords(impl_.Kak4(m).v, atol);
    }

    // ---- 块合并 ----

    // _verified_euler_ops_uncached：ops；False = Python 返回 None；None = 输入含非有限值（交给 Python）。
    py::object VerifiedEuler(const ComplexArray& u, double atol, int euler) const {
        try {
            const auto ops = impl_.VerifiedEuler(ToMatrix<4>(u), atol, euler);
            if (!ops) {
                return py::bool_(false);
            }
            return conv_.Convert(*ops, py::str("?"));
        } catch (const SynthesisFailed&) {
            return py::none();
        }
    }

    // 块的 4×4 酉矩阵（见 KakSynthesizer::BlockUnitary）。
    py::array_t<std::complex<double>> BlockUnitary(const py::list& mats, const std::vector<int>& slots) const {
        if (mats.size() != slots.size()) {
            throw std::invalid_argument("one slot per matrix");
        }
        std::vector<ComplexArray> keep;
        keep.reserve(mats.size());
        std::vector<KakSynthesizer::BlockMember> members;
        members.reserve(mats.size());
        for (std::size_t i = 0; i < mats.size(); ++i) {
            keep.push_back(mats[i].cast<ComplexArray>());
            const int slot = slots[i];
            const py::ssize_t expected = (slot == 1 || slot == 2) ? 4 : 16;
            if (slot < 0 || slot > 4 || keep.back().size() != expected) {
                throw std::invalid_argument("bad block member");
            }
            members.push_back({reinterpret_cast<const Cx*>(keep.back().data()), slot});
        }
        return FromMatrix(impl_.BlockUnitary(members));
    }

    // 单比特门串的矩阵：u = I；对每个门 u = m @ u。
    py::array_t<std::complex<double>> RunUnitary(const py::list& mats) const {
        M2 u{Cx{1.0, 0.0}, Cx{0.0, 0.0}, Cx{0.0, 0.0}, Cx{1.0, 0.0}};
        for (const auto item : mats) {
            u = detmath::MatMul(ToMatrix<4>(item.cast<ComplexArray>()), u);
        }
        return FromMatrix(u);
    }

    // _mapping._swap_residual 的数值部分：(A, B) 或 None。
    py::object SwapResidual(const ComplexArray& u, double tol) const {
        const auto res = KakSynthesizer::SwapResidual(ToMatrix<16>(u), tol);
        if (!res) {
            return py::none();
        }
        return py::make_tuple(FromMatrix((*res)[0]), FromMatrix((*res)[1]));
    }

    // ---- 单比特合成 ----
    int AddOneQubitBasis(bool rz, bool h, bool x, bool sx, bool sxdg, bool rx, bool ry,
                         bool u3, const py::dict& words, const std::vector<int>& named) {
        if (!one_qubit_) {
            one_qubit_ = std::make_unique<OneQubitSynthesizer>();
        }
        OneQubitBasis b;
        b.rz = rz;
        b.h = h;
        b.x = x;
        b.sx = sx;
        b.sxdg = sxdg;
        b.rx = rx;
        b.ry = ry;
        b.u3 = u3;
        for (const auto item : words) {
            const int k = item.first.cast<int>();
            if (k < -4 || k > 4) {
                throw std::invalid_argument("Clifford+T word index out of range");
            }
            b.has_word[static_cast<std::size_t>(k + 4)] = true;
            b.words[static_cast<std::size_t>(k + 4)] = item.second.cast<std::vector<int>>();
        }
        for (int g : named) {
            if (g < 0 || g >= kNumFixedGates) {
                throw std::invalid_argument("bad fixed gate index");
            }
        }
        b.named = named;
        one_qubit_bases_.push_back(std::move(b));
        return static_cast<int>(one_qubit_bases_.size()) - 1;
    }

    // _synthesize_one_qubit：(0, [(kind, index, p0, p1, p2), ...])；(1, None) = Python 返回 None；
    // (2, None) = 输入含非有限值（交给 Python）。
    py::tuple SynthesizeOneQubit(const ComplexArray& m, int basis) const {
        if (!one_qubit_ || basis < 0 || static_cast<std::size_t>(basis) >= one_qubit_bases_.size()) {
            throw std::invalid_argument("unknown single-qubit basis");
        }
        try {
            const auto gates = one_qubit_->Synthesize(ToMatrix<4>(m), one_qubit_bases_[static_cast<std::size_t>(basis)]);
            if (!gates) {
                return py::make_tuple(1, py::none());
            }
            py::list out(gates->size());
            for (std::size_t i = 0; i < gates->size(); ++i) {
                const Gate1& g = (*gates)[i];
                out[i] = py::make_tuple(g.kind, g.index, g.p0, g.p1, g.p2);
            }
            return py::make_tuple(0, out);
        } catch (const std::domain_error&) {
            return py::make_tuple(2, py::none());
        }
    }

    py::tuple ZyzAngles(const ComplexArray& m) {
        if (!one_qubit_) {
            one_qubit_ = std::make_unique<OneQubitSynthesizer>();
        }
        const auto a = one_qubit_->ZyzAngles(ToMatrix<4>(m));
        return py::make_tuple(a[0], a[1], a[2]);
    }

    // ---- 测试与跨平台指纹用 ----
    py::tuple Kak4(const ComplexArray& u) const {
        const Kak k = impl_.Kak4(ToMatrix<16>(u));
        return py::make_tuple(k.phase, FromMatrix(k.k1), py::make_tuple(k.v[0], k.v[1], k.v[2]), FromMatrix(k.k2));
    }

    py::tuple FactorLocal(const ComplexArray& k) const {
        const auto f = impl_.FactorLocal(ToMatrix<16>(k));
        return py::make_tuple(FromMatrix(f[0]), FromMatrix(f[1]));
    }

    py::list EulerOps(const ComplexArray& m, int qubit, int basis, double atol) const {
        std::vector<Op> ops;
        impl_.EulerOps(ToMatrix<4>(m), qubit, basis, atol, &ops);
        return conv_.Convert(ops, py::str("?"));
    }

    py::array_t<std::complex<double>> Kron2(const ComplexArray& a, const ComplexArray& b) const {
        return FromMatrix(KakSynthesizer::Kron2(ToMatrix<4>(a), ToMatrix<4>(b)));
    }

    py::array_t<std::complex<double>> AxisRot(const ComplexArray& pauli, double phi) const {
        return FromMatrix(impl_.AxisRot(ToMatrix<4>(pauli), phi));
    }

    double AxisAngle(const ComplexArray& w, const ComplexArray& pauli) const {
        return KakSynthesizer::AxisAngle(ToMatrix<4>(w), ToMatrix<4>(pauli));
    }

 private:

    SynthOptions Options(int entangler, int euler, double atol, double verify_atol, bool swap_face_template,
                         bool mirrored) const {
        if (entangler < 0 || static_cast<std::size_t>(entangler) >= names_.size()) {
            throw std::invalid_argument("unknown entangler");
        }
        SynthOptions opt;
        opt.entangler = entangler;
        opt.euler = euler;
        opt.atol = atol;
        opt.verify_atol = verify_atol;
        opt.swap_face_template = swap_face_template;
        opt.mirrored_stabilizers = mirrored;
        return opt;
    }

    const py::str& Name(int entangler) const {
        return names_.at(static_cast<std::size_t>(entangler));
    }

    KakSynthesizer impl_;
    OpConverter conv_;
    std::vector<py::str> names_;
    std::unique_ptr<OneQubitSynthesizer> one_qubit_;
    std::vector<OneQubitBasis> one_qubit_bases_;
};
// gate_matrix(gate) 的确定性实现：未知门名或参数个数不对时返回 None（由 Python 端报错）。
py::object GateMatrixPy(const std::string& name, const std::vector<double>& params) {
    Cx out[64];
    int dim = 0;
    if (!GateMatrix(name, params.data(), static_cast<int>(params.size()), out, &dim)) {
        return py::none();
    }
    py::array_t<std::complex<double>> arr({static_cast<py::ssize_t>(dim), static_cast<py::ssize_t>(dim)});
    std::memcpy(arr.mutable_data(), out, sizeof(Cx) * static_cast<std::size_t>(dim * dim));
    return std::move(arr);
}

py::array_t<std::complex<double>> FromM8(const M8& m) {
    py::array_t<std::complex<double>> out({static_cast<py::ssize_t>(8), static_cast<py::ssize_t>(8)});
    std::memcpy(out.mutable_data(), m.data(), sizeof(M8));
    return out;
}

// _three_qubit 的块酉矩阵：u = I；对每个 (矩阵, 局部比特) 左乘。
py::array_t<std::complex<double>> ThreeQubitUnitaryPy(const py::list& mats,
                                                      const std::vector<std::vector<int>>& qubits) {
    if (mats.size() != qubits.size()) {
        throw std::invalid_argument("one qubit tuple per matrix");
    }
    std::vector<ComplexArray> keep;
    keep.reserve(mats.size());
    std::vector<LocalGate> gates;
    gates.reserve(mats.size());
    for (std::size_t i = 0; i < mats.size(); ++i) {
        keep.push_back(mats[i].cast<ComplexArray>());
        const auto dim = static_cast<py::ssize_t>(1) << qubits[i].size();
        if (qubits[i].empty() || qubits[i].size() > 3 || keep.back().size() != dim * dim) {
            throw std::invalid_argument("matrix size does not match its qubits");
        }
        gates.push_back({reinterpret_cast<const Cx*>(keep.back().data()), qubits[i]});
    }
    return FromM8(ThreeQubitUnitary(gates));
}

M8 ToM8(const ComplexArray& a) {
    if (a.size() != 64) {
        throw std::invalid_argument("expected an 8x8 matrix");
    }
    M8 m;
    std::memcpy(m.data(), a.data(), sizeof(M8));
    return m;
}

py::object DemultiplexPy(const ComplexArray& a, const ComplexArray& b) {
    const auto r = Demultiplex(ToMatrix<16>(a), ToMatrix<16>(b));
    if (!r) {
        return py::none();
    }
    return py::make_tuple(FromMatrix(r->v), FromMatrix(r->w),
                          py::make_tuple(r->phases[0], r->phases[1], r->phases[2], r->phases[3]));
}

py::object CosSinPy(const ComplexArray& u) {
    const auto r = CosSinDecompose(ToM8(u));
    if (!r) {
        return py::none();
    }
    py::array_t<double> theta(4);
    std::memcpy(theta.mutable_data(), r->theta.data(), sizeof(double) * 4);
    return py::make_tuple(py::make_tuple(FromMatrix(r->u1), FromMatrix(r->u2)), theta,
                          py::make_tuple(FromMatrix(r->v1h), FromMatrix(r->v2h)));
}

}  // namespace

void BindSynthesis(py::module_& m) {
    py::class_<PyKakSynthesizer>(m, "_KakSynthesizer", py::module_local(),
                                 "Two-qubit KAK synthesis, single-qubit synthesis and block products on the "
                                 "deterministic numerics layer (bit-identical on all platforms).")
        .def(py::init<>())
        .def("add_entangler", &PyKakSynthesizer::AddEntangler, py::arg("name"), py::arg("matrix"), py::arg("before0"),
             py::arg("before1"), py::arg("after0"), py::arg("after1"), py::arg("symmetric"))
        .def("synthesize", &PyKakSynthesizer::Synthesize, py::arg("u"), py::arg("entangler"), py::arg("euler"),
             py::arg("atol"), py::arg("verify_atol"), py::arg("swap_face_template"), py::arg("mirrored"))
        .def("approximate", &PyKakSynthesizer::Approximate, py::arg("u"), py::arg("max_infidelity"), py::arg("atol"),
             py::arg("verify_atol"), py::arg("max_entanglers"), py::arg("entangler"), py::arg("euler"),
             py::arg("swap_face_template"), py::arg("mirrored"), py::arg("want_exact"))
        .def("swap_face", &PyKakSynthesizer::SwapFace, py::arg("u"), py::arg("entangler"), py::arg("euler"),
             py::arg("verify_atol"), py::arg("mirrored"))
        .def("num_cz", &PyKakSynthesizer::NumCz, py::arg("u"), py::arg("atol"))
        .def("add_one_qubit_basis", &PyKakSynthesizer::AddOneQubitBasis, py::arg("rz"), py::arg("h"), py::arg("x"),
             py::arg("sx"), py::arg("sxdg"), py::arg("rx"), py::arg("ry"), py::arg("u3"), py::arg("words"),
             py::arg("named"))
        .def("synthesize_one_qubit", &PyKakSynthesizer::SynthesizeOneQubit, py::arg("m"), py::arg("basis"))
        .def("zyz_angles", &PyKakSynthesizer::ZyzAngles, py::arg("m"))
        .def("verified_euler", &PyKakSynthesizer::VerifiedEuler, py::arg("u"), py::arg("atol"), py::arg("euler"))
        .def("block_unitary", &PyKakSynthesizer::BlockUnitary, py::arg("mats"), py::arg("slots"))
        .def("run_unitary", &PyKakSynthesizer::RunUnitary, py::arg("mats"))
        .def("swap_residual", &PyKakSynthesizer::SwapResidual, py::arg("u"), py::arg("tol"))
        .def("kak4", &PyKakSynthesizer::Kak4, py::arg("u"))
        .def("factor_local", &PyKakSynthesizer::FactorLocal, py::arg("k"))
        .def("euler_ops", &PyKakSynthesizer::EulerOps, py::arg("m"), py::arg("qubit"), py::arg("basis"),
             py::arg("atol"))
        .def("kron2", &PyKakSynthesizer::Kron2, py::arg("a"), py::arg("b"))
        .def("axis_rot", &PyKakSynthesizer::AxisRot, py::arg("pauli"), py::arg("phi"))
        .def("axis_angle", &PyKakSynthesizer::AxisAngle, py::arg("w"), py::arg("pauli"));
    m.def("_three_qubit_unitary", &ThreeQubitUnitaryPy, py::arg("mats"), py::arg("qubits"),
          "8x8 unitary of gates on local qubits (0 = high), deterministic products.");
    m.def(
        "_three_qubit_distance",
        [](const ComplexArray& u, const ComplexArray& v) { return ThreeQubitDistance(ToM8(u), ToM8(v)); },
        py::arg("u"), py::arg("v"), "Phase-aligned max distance of two 8x8 matrices (inf if orthogonal).");
    m.def("_three_qubit_demultiplex", &DemultiplexPy, py::arg("a"), py::arg("b"),
          "(V, W, phases) with A = V D W, B = V D^dagger W, or None (deterministic eigendecomposition).");
    m.def("_three_qubit_cossin", &CosSinPy, py::arg("u"),
          "((u1, u2), theta, (v1h, v2h)) as scipy.linalg.cossin(u, p=4, q=4, separate=True), or None.");
    m.def("_gate_matrix", &GateMatrixPy, py::arg("name"), py::arg("params"),
          "Deterministic matrix of a standard gate (None for an unknown name or a wrong parameter count).");
}

}  // namespace quspect::native::pybind

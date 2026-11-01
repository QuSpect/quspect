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

#include "quspect_native/synthesis/kak_synthesis.hpp"
#include "quspect_native/synthesis/numpy_compat.hpp"
#include "quspect_native/synthesis/numpy_euler.hpp"
#include "quspect_native/synthesis/one_qubit_synthesis.hpp"

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

std::uintptr_t Address(const py::dict& fns, const char* name) {
    if (!fns.contains(name)) {
        return 0;
    }
    return fns[name].cast<std::uintptr_t>();
}

MulMode ToMulMode(int v) {
    if (v < 0 || v > 2) {
        throw std::invalid_argument("bad multiplication mode");
    }
    return static_cast<MulMode>(v);
}

ArithModes ToModes(const py::dict& d) {
    ArithModes m;
    m.arr_scalar = ToMulMode(d["arr_scalar"].cast<int>());
    m.scalar_arr = ToMulMode(d["scalar_arr"].cast<int>());
    m.kron = ToMulMode(d["kron"].cast<int>());
    m.np_scalar = ToMulMode(d["np_scalar"].cast<int>());
    m.linalg = ToMulMode(d["linalg"].cast<int>());
    m.py = ToMulMode(d["py"].cast<int>());
    m.py_mixed = d["py_mixed"].cast<bool>();
    m.np_div = d["np_div"].cast<int>() != 0 ? DivMode::kFma : DivMode::kPlain;
    m.py_exp_joint = d["py_exp_joint"].cast<bool>();
    m.np_exp_joint = d["np_exp_joint"].cast<bool>();
    return m;
}

LinalgFunctions ToFunctions(const py::dict& fns, bool ilp64) {
    LinalgFunctions f;
    f.ilp64 = ilp64;
    f.zgemm = Address(fns, "zgemm");
    f.zsyrk = Address(fns, "zsyrk");
    f.dgemv = Address(fns, "dgemv");
    f.zdotc_sub = Address(fns, "zdotc_sub");
    f.zgesdd = Address(fns, "zgesdd");
    f.dsyevd = Address(fns, "dsyevd");
    f.zgetrf = Address(fns, "zgetrf");
    f.dgetrf = Address(fns, "dgetrf");
    return f;
}

// numpy 语义部分（NumpyEuler）用到的 Python 常量矩阵（与 two_qubit.py 的模块常量逐位相同）。
KakConstants ToNumpyConstants(const py::dict& d) {
    KakConstants c = KakConstants::Make();
    auto m2 = [&](const char* key) { return ToMatrix<4>(d[key].cast<ComplexArray>()); };
    c.i2 = m2("i2");
    c.x = m2("x");
    c.h = m2("h");
    c.sx = m2("sx");
    return c;
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

// 两比特 KAK 合成（确定性数值层，全平台逐位一致）与 numpy 语义的块合并辅助函数。
//
// functions / modes 为 None 时只提供 KAK 合成；否则同时构建 NumpyEuler（verified_euler、block_unitary、
// run_unitary、单比特合成与 KakLinalg），它们与本平台的 numpy 逐位一致。
class PyKakSynthesizer {
 public:
    PyKakSynthesizer(const py::object& fns, bool ilp64, const py::object& modes, const py::dict& constants) {
        if (!fns.is_none() && !modes.is_none()) {
            numpy_ = std::make_unique<NumpyEuler>(ToFunctions(fns.cast<py::dict>(), ilp64),
                                                  ToModes(modes.cast<py::dict>()), ToNumpyConstants(constants));
        }
    }

    bool HasNumpy() const {
        return numpy_ != nullptr;
    }

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

    // ---- numpy 语义部分（HasNumpy() 为 false 时抛出 RuntimeError）----

    // _verified_euler_ops_uncached：ops；False = Python 返回 None；None = 交给 Python。
    py::object VerifiedEuler(const ComplexArray& u, double atol, int euler) const {
        const NumpyEuler& np = Numpy();
        last_fallback_.clear();
        try {
            const auto ops = np.VerifiedEuler(ToMatrix<4>(u), atol, euler);
            if (!ops) {
                return py::bool_(false);
            }
            return conv_.Convert(*ops, py::str("?"));
        } catch (const Fallback& e) {
            last_fallback_ = e.what();
            return py::none();
        }
    }

    // 块的 4×4 酉矩阵（见 NumpyEuler::BlockUnitary）。
    py::array_t<std::complex<double>> BlockUnitary(const py::list& mats, const std::vector<int>& slots,
                                                   const ComplexArray& swap4) const {
        const NumpyEuler& np = Numpy();
        if (mats.size() != slots.size()) {
            throw std::invalid_argument("one slot per matrix");
        }
        std::vector<ComplexArray> keep;
        keep.reserve(mats.size());
        std::vector<NumpyEuler::BlockMember> members;
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
        return FromMatrix(np.BlockUnitary(members, ToMatrix<16>(swap4)));
    }

    // 与 BlockUnitary / RunUnitary 相同，但先核对每个矩阵都是 C 连续的 complex128 数组
    // （numpy 的 BLAS 调用取决于操作数的布局；与 Python 端 m.dtype == complex128 and
    // m.flags.c_contiguous 的判断相同，run 还要求形状为 (2, 2)），否则返回 None。
    py::object BlockUnitaryStrict(const py::list& mats, const std::vector<int>& slots,
                                  const ComplexArray& swap4) const {
        for (const auto item : mats) {
            if (!StrictArray::check_(item)) {
                return py::none();
            }
        }
        return BlockUnitary(mats, slots, swap4);
    }
    py::object RunUnitaryStrict(const py::list& mats) const {
        for (const auto item : mats) {
            if (!StrictArray::check_(item)) {
                return py::none();
            }
            const auto arr = py::reinterpret_borrow<py::array>(item);
            if (arr.ndim() != 2 || arr.shape(0) != 2 || arr.shape(1) != 2) {
                return py::none();
            }
        }
        return RunUnitary(mats);
    }

    // 单比特门串的矩阵：u = I；对每个门 u = m @ u（numpy 的 2×2 乘法）。
    py::array_t<std::complex<double>> RunUnitary(const py::list& mats) const {
        const NumpyLinalg& la = Numpy().linalg();
        M2 u{Cx{1.0, 0.0}, Cx{0.0, 0.0}, Cx{0.0, 0.0}, Cx{1.0, 0.0}};
        for (const auto item : mats) {
            const M2 m = ToMatrix<4>(item.cast<ComplexArray>());
            M2 next;
            la.MatMul(2, View(m), View(u), next.data());
            u = next;
        }
        return FromMatrix(u);
    }

    // ---- 单比特合成（numpy 语义）----
    int AddOneQubitBasis(const py::list& fixed, bool rz, bool h, bool x, bool sx, bool sxdg, bool rx, bool ry,
                         bool u3, const py::dict& words, const std::vector<int>& named) {
        if (!one_qubit_) {
            if (fixed.size() != kNumFixedGates) {
                throw std::invalid_argument("need the matrices of the 10 fixed gates");
            }
            std::array<M2, kNumFixedGates> mats;
            for (std::size_t i = 0; i < mats.size(); ++i) {
                mats[i] = ToMatrix<4>(fixed[i].cast<ComplexArray>());
            }
            one_qubit_ = std::make_unique<OneQubitSynthesizer>(Numpy().linalg().modes(), mats);
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
    // (2, None) = 交给 Python。
    py::tuple SynthesizeOneQubit(const ComplexArray& m, int basis) const {
        last_fallback_.clear();
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
        } catch (const Fallback& e) {
            last_fallback_ = e.what();
            return py::make_tuple(2, py::none());
        }
    }

    py::tuple ZyzAngles(const ComplexArray& m) const {
        if (!one_qubit_) {
            throw std::invalid_argument("no single-qubit basis registered");
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

    // numpy 语义的乘法与行列式（探测 / 对拍用）。
    py::array_t<std::complex<double>> MatMul(const ComplexArray& a, const ComplexArray& b, bool a_t, bool b_t) const {
        const auto n = static_cast<int>(a.shape(0));
        std::vector<Cx> out(static_cast<std::size_t>(n * n));
        const Cx* pa = reinterpret_cast<const Cx*>(a.data());
        const Cx* pb = reinterpret_cast<const Cx*>(b.data());
        const CView va = a_t ? CView{pa, 1, n} : CView{pa, n, 1};
        const CView vb = b_t ? CView{pb, 1, n} : CView{pb, n, 1};
        Numpy().linalg().MatMul(n, va, vb, out.data());
        py::array_t<std::complex<double>> res({n, n});
        std::memcpy(res.mutable_data(), out.data(), out.size() * sizeof(Cx));
        return res;
    }

    std::complex<double> Det(const ComplexArray& a) const {
        const auto n = static_cast<int>(a.shape(0));
        const Cx* pa = reinterpret_cast<const Cx*>(a.data());
        const Cx d = Numpy().linalg().Det(n, CView{pa, n, 1});
        return {d.re, d.im};
    }

    // 最近一次退回 Python 的原因（numpy 语义部分，诊断用）。
    const std::string& LastFallback() const {
        return last_fallback_;
    }

    // numpy 语义的线性代数（单比特门最小化等其他原生实现复用）；没有 numpy 部分时为 nullptr。
    const synthesis::NumpyLinalg* Linalg() const {
        return numpy_ ? &numpy_->linalg() : nullptr;
    }

 private:
    const NumpyEuler& Numpy() const {
        if (!numpy_) {
            throw std::runtime_error("numpy-semantics operations are not available on this synthesizer");
        }
        return *numpy_;
    }

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
    std::unique_ptr<NumpyEuler> numpy_;
    OpConverter conv_;
    std::vector<py::str> names_;
    mutable std::string last_fallback_;
    std::unique_ptr<OneQubitSynthesizer> one_qubit_;
    std::vector<OneQubitBasis> one_qubit_bases_;
};
// 逐元素运算的各种运算顺序（探测 numpy / CPython 的实际语义用）：op 0 = 乘法（mode 0–2），1 = numpy
// 复数除法（mode 0–1），2 = CPython complex * float（mode 0–2 为完整复数乘法，3 = 混合运算）。
py::array_t<std::complex<double>> Arith(int op, int mode, const ComplexArray& a, const ComplexArray& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("operand sizes differ");
    }
    const auto n = static_cast<std::size_t>(a.size());
    const Cx* pa = reinterpret_cast<const Cx*>(a.data());
    const Cx* pb = reinterpret_cast<const Cx*>(b.data());
    py::array_t<std::complex<double>> out(static_cast<py::ssize_t>(n));
    Cx* po = reinterpret_cast<Cx*>(out.mutable_data());
    for (std::size_t i = 0; i < n; ++i) {
        switch (op) {
            case 0:
                po[i] = Mul(ToMulMode(mode), pa[i], pb[i]);
                break;
            case 1:
                po[i] = NpDiv(mode != 0 ? DivMode::kFma : DivMode::kPlain, pa[i], pb[i]);
                break;
            default: {
                ArithModes m;
                m.py_mixed = mode == 3;
                m.py = ToMulMode(mode == 3 ? 0 : mode);
                po[i] = PyMulReal(m, pa[i], pb[i].re);
                break;
            }
        }
    }
    return out;
}

// CPython / libm 标量函数（对拍用）：0 = cmath.sqrt，1 = cmath.exp（分别求 sin、cos），2 = cmath.phase，
// 3 = math.atan2(im, re)，4 = libm atan2(im, re)，5 = (cos(im), sin(im))，6 = abs，7 = cmath.exp（合并的 sincos），
// 8 = 合并的 sincos 求 (cos(im), sin(im))。
py::array_t<std::complex<double>> Scalar(int fn, const ComplexArray& a) {
    const auto n = static_cast<std::size_t>(a.size());
    const Cx* pa = reinterpret_cast<const Cx*>(a.data());
    py::array_t<std::complex<double>> out(static_cast<py::ssize_t>(n));
    Cx* po = reinterpret_cast<Cx*>(out.mutable_data());
    for (std::size_t i = 0; i < n; ++i) {
        const Cx& z = pa[i];
        switch (fn) {
            case 0:
                po[i] = PySqrt(z);
                break;
            case 1:
            case 7: {
                ArithModes m;
                m.py_exp_joint = fn == 7;
                po[i] = PyExp(m, z);
                break;
            }
            case 2:
                po[i] = Cx{PyPhase(z), 0.0};
                break;
            case 3:
                po[i] = Cx{PyAtan2(z.im, z.re), 0.0};
                break;
            case 4:
                po[i] = Cx{std::atan2(z.im, z.re), 0.0};
                break;
            case 5:
                po[i] = Cx{SeparateCos(z.im), SeparateSin(z.im)};
                break;
            case 8: {
                double s;
                double c;
                JointSinCos(z.im, &s, &c);
                po[i] = Cx{c, s};
                break;
            }
            default:
                po[i] = Cx{PyAbs(z), 0.0};
                break;
        }
    }
    return out;
}

}  // namespace

void BindSynthesis(py::module_& m) {
    py::class_<PyKakSynthesizer>(m, "_KakSynthesizer", py::module_local(),
                                 "Two-qubit KAK synthesis on the deterministic numerics layer (bit-identical on all "
                                 "platforms), plus numpy-semantics block helpers when functions/modes are given.")
        .def(py::init<const py::object&, bool, const py::object&, const py::dict&>(), py::arg("functions"),
             py::arg("ilp64"), py::arg("modes"), py::arg("constants"))
        .def_property_readonly("has_numpy", &PyKakSynthesizer::HasNumpy)
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
        .def("add_one_qubit_basis", &PyKakSynthesizer::AddOneQubitBasis, py::arg("fixed"), py::arg("rz"), py::arg("h"),
             py::arg("x"), py::arg("sx"), py::arg("sxdg"), py::arg("rx"), py::arg("ry"), py::arg("u3"),
             py::arg("words"), py::arg("named"))
        .def("synthesize_one_qubit", &PyKakSynthesizer::SynthesizeOneQubit, py::arg("m"), py::arg("basis"))
        .def("zyz_angles", &PyKakSynthesizer::ZyzAngles, py::arg("m"))
        .def("verified_euler", &PyKakSynthesizer::VerifiedEuler, py::arg("u"), py::arg("atol"), py::arg("euler"))
        .def("block_unitary", &PyKakSynthesizer::BlockUnitary, py::arg("mats"), py::arg("slots"), py::arg("swap4"))
        .def("run_unitary", &PyKakSynthesizer::RunUnitary, py::arg("mats"))
        .def("block_unitary_strict", &PyKakSynthesizer::BlockUnitaryStrict, py::arg("mats"), py::arg("slots"),
             py::arg("swap4"))
        .def("run_unitary_strict", &PyKakSynthesizer::RunUnitaryStrict, py::arg("mats"))
        .def("kak4", &PyKakSynthesizer::Kak4, py::arg("u"))
        .def("factor_local", &PyKakSynthesizer::FactorLocal, py::arg("k"))
        .def("euler_ops", &PyKakSynthesizer::EulerOps, py::arg("m"), py::arg("qubit"), py::arg("basis"),
             py::arg("atol"))
        .def("kron2", &PyKakSynthesizer::Kron2, py::arg("a"), py::arg("b"))
        .def("axis_rot", &PyKakSynthesizer::AxisRot, py::arg("pauli"), py::arg("phi"))
        .def("axis_angle", &PyKakSynthesizer::AxisAngle, py::arg("w"), py::arg("pauli"))
        .def("matmul", &PyKakSynthesizer::MatMul, py::arg("a"), py::arg("b"), py::arg("a_t"), py::arg("b_t"))
        .def("det", &PyKakSynthesizer::Det, py::arg("a"))
        .def_property_readonly("last_fallback", &PyKakSynthesizer::LastFallback);
    m.def("_synthesis_arith", &Arith, py::arg("op"), py::arg("mode"), py::arg("a"), py::arg("b"),
          "Elementwise complex arithmetic in one of the candidate evaluation orders (semantics probes).");
    m.def("_synthesis_scalar", &Scalar, py::arg("fn"), py::arg("a"),
          "CPython / libm scalar functions as reproduced by the native synthesis (semantics probes).");
}

const synthesis::NumpyLinalg* KakLinalg(py::handle obj) {
    if (!py::isinstance<PyKakSynthesizer>(obj)) {
        return nullptr;
    }
    return obj.cast<const PyKakSynthesizer&>().Linalg();
}

}  // namespace quspect::native::pybind

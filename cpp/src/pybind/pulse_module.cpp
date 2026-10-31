// quspect._pulse：控制包络的 pybind11 绑定，Python 接口与 envelopes-qc 0.1.5 的
// envelopes.envelopes_cpp 保持一致（类名、签名、文档字符串和行为）。

#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pulse_docstrings.hpp"
#include "quspect_native/pulse/envelopes.hpp"

namespace py = pybind11;
using namespace quspect::native::pulse;

namespace {

template <class T>
using InArray = py::array_t<T, py::array::forcecast>;

// pybind11 会把 None 转成空的 shared_ptr；直接拒绝，避免构造出含空节点的树。
const EnvPtr& Required(const EnvPtr& env) {
    if (!env) {
        throw py::type_error("envelope argument must not be None");
    }
    return env;
}

template <class T>
std::vector<T> ToVector(const InArray<T>& a) {
    if (!(a.flags() & py::array::c_style)) {
        throw std::runtime_error("NDArray must be c contiguous. Warp the NDArray with np.ascontiguousarray");
    }
    const T* p = a.data();
    return std::vector<T>(p, p + a.size());
}

// 零拷贝的 numpy 视图，由 capsule 持有解码缓冲区。
template <class T>
py::array AsNumpy(const WavePtr<T>& wave, long long size) {
    if (size < 0) {
        throw py::value_error("negative dimensions are not allowed");
    }
    auto* keep = new WavePtr<T>(wave);
    py::capsule owner(keep, [](void* p) { delete static_cast<WavePtr<T>*>(p); });
    return py::array_t<T>({static_cast<py::ssize_t>(size)}, {static_cast<py::ssize_t>(sizeof(T))}, wave->data(), owner);
}

std::tuple<double, py::array> ToPython(const Decoded& d) {
    if (d.complex) {
        return {d.t, AsNumpy(d.complex, d.size)};
    }
    return {d.t, AsNumpy(d.real, d.size)};
}

EnvPtr MakeCustom(double start, double dt, const py::array& data) {
    if (data.ndim() != 1) {
        throw std::runtime_error("Custom envelope data must be a 1-D array.");
    }
    if (data.size() == 0) {
        throw std::runtime_error("Custom envelope data must not be empty.");
    }
    const char kind = data.dtype().kind();
    if (kind != 'c' && kind != 'f' && kind != 'i' && kind != 'u' && kind != 'b') {
        throw std::runtime_error("Custom envelope data must be a numeric 1-D array.");
    }
    if (!(dt > 0)) {
        throw std::runtime_error("Custom envelope dt must be positive.");
    }
    // float32/64 与 complex64/128 直接读取原数组，必须 C 连续；其他 dtype 先转换成新的连续副本。
    const bool direct = (kind == 'f' || kind == 'c')
                        && (data.itemsize() == 4 || data.itemsize() == 8 || data.itemsize() == 16);
    if (direct && !(data.flags() & py::array::c_style)) {
        throw std::runtime_error("NDArray must be c contiguous. Warp the NDArray with np.ascontiguousarray");
    }
    if (kind == 'c') {
        auto a = py::array_t<CFloat, py::array::c_style | py::array::forcecast>::ensure(data);
        return std::make_shared<Custom>(start, dt,
                                        std::make_shared<std::vector<CFloat>>(a.data(), a.data() + a.size()));
    }
    auto a = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(data);
    return std::make_shared<Custom>(start, dt, std::make_shared<std::vector<float>>(a.data(), a.data() + a.size()));
}

template <class T, class Cls>
void AddPickle(Cls& cls) {
    cls.def(py::pickle([](const EnvPtr& self) { return py::bytes(Serialize(self)); },
                       [](const py::bytes& state) {
                           auto result = std::dynamic_pointer_cast<T>(Deserialize(state));
                           if (!result) {
                               throw std::runtime_error("Pickle state does not match the envelope type.");
                           }
                           return result;
                       }));
}

template <class T>
using EnvClass = py::class_<T, Envelope, std::shared_ptr<T>>;

template <class T>
EnvClass<T> EnvelopeClass(py::module_& m, const char* name, const char* doc) {
    EnvClass<T> cls(m, name, doc);
    AddPickle<T>(cls);
    return cls;
}

}  // namespace

PYBIND11_MODULE(_pulse, m) {
    m.doc() = docs::kModule;

    py::class_<WaveCache>(m, "WaveCache", docs::kWaveCacheClass)
        .def(py::init<double>(), py::arg("resolution"))
        .def("resolution", &WaveCache::resolution, docs::kWaveCacheResolution)
        .def("__repr__", &WaveCache::ToString);

    // 运算符按普通方法绑定（与原实现一致）：操作数类型不匹配时抛出 TypeError，而不是返回 NotImplemented。
    py::class_<AbstractEnvelope, std::shared_ptr<AbstractEnvelope>>(m, "AbstractEnvelope")
        .def("start", &AbstractEnvelope::Start)
        .def("end", &AbstractEnvelope::End)
        .def("is_complex", &AbstractEnvelope::IsComplex)
        .def("copy", [](const EnvPtr& self) { return Copy(self); })
        .def("__copy__", [](const EnvPtr& self) { return Copy(self); })
        .def("__add__", [](const EnvPtr& self, double c) { return Add(self, c); })
        .def("__add__", [](const EnvPtr& self, const EnvPtr& other) { return Add(self, Required(other)); })
        .def("__radd__", [](const EnvPtr& self, double c) { return Add(self, c); })
        .def("__radd__", [](const EnvPtr& self, const EnvPtr& other) { return Add(Required(other), self); })
        .def("__iadd__", [](const EnvPtr& self, const EnvPtr& other) { return InplaceAdd(self, Required(other)); })
        .def("__mul__", [](const EnvPtr& self, float c) { return Multiply(self, c); })
        .def("__mul__", [](const EnvPtr& self, CFloat c) { return Multiply(self, c); })
        .def("__rmul__", [](const EnvPtr& self, float c) { return Multiply(self, c); })
        .def("__rmul__", [](const EnvPtr& self, CFloat c) { return Multiply(self, c); })
        .def("__rshift__", [](const EnvPtr& self, double dt) { return Shift(self, dt); })
        .def("__repr__", &AbstractEnvelope::ToString);

    py::class_<Envelope, AbstractEnvelope, std::shared_ptr<Envelope>>(m, "Envelope");

    EnvelopeClass<Gaussian>(m, "Gaussian", docs::kGaussianClass)
        .def(py::init<double, double, float>(), py::arg("t0"), py::arg("w"), py::arg("amp"), docs::kGaussianInit);
    EnvelopeClass<GaussianDrag>(m, "GaussianDRAG", docs::kGaussianDRAGClass)
        .def(py::init<double, double, float, double, double, double>(), py::arg("t0"), py::arg("w"), py::arg("amp"),
             py::arg("coef"), py::arg("df"), py::arg("phase"), docs::kGaussianDRAGInit);
    EnvelopeClass<CosineDrag>(m, "CosineDRAG", docs::kCosineDRAGClass)
        .def(py::init<double, double, float, double, double, double>(), py::arg("t0"), py::arg("w"), py::arg("amp"),
             py::arg("coef"), py::arg("df"), py::arg("phase"), docs::kCosineDRAGInit);
    EnvelopeClass<Triangle>(m, "Triangle", docs::kTriangleClass)
        .def(py::init<double, double, float, bool>(), py::arg("t0"), py::arg("tlen"), py::arg("amp"), py::arg("fall"),
             docs::kTriangleInit);
    EnvelopeClass<Rect>(m, "Rect", docs::kRectClass)
        .def(py::init<double, double, float>(), py::arg("t0"), py::arg("tlen"), py::arg("amp"), docs::kRectInit);
    EnvelopeClass<Flattop>(m, "Flattop", docs::kFlattopClass)
        .def(py::init<double, double, double, double, float>(), py::arg("t0"), py::arg("tlen"), py::arg("w_left"),
             py::arg("w_right"), py::arg("amp"), docs::kFlattopInit);
    EnvelopeClass<RippleRect>(m, "RippleRect", docs::kRippleRectClass)
        .def(py::init<double, double, float, double, double, double, double, double>(), py::arg("t0"), py::arg("tlen"),
             py::arg("amp"), py::arg("w"), py::arg("ripple0"), py::arg("ripple1"), py::arg("ripple2"),
             py::arg("ripple3"), docs::kRippleRectInit);
    EnvelopeClass<PowerSineCosine>(m, "PowerSineCosine", docs::kPowerSineCosineClass)
        .def(py::init<double, double, float, int, int, double, double>(), py::arg("t0"), py::arg("tlen"),
             py::arg("amp"), py::arg("m"), py::arg("n"), py::arg("k"), py::arg("phase"), docs::kPowerSineCosineInit);
    EnvelopeClass<GaussianXPower>(m, "GaussianXPower", docs::kGaussianXPowerClass)
        .def(py::init<double, double, float, int>(), py::arg("t0"), py::arg("w"), py::arg("amp"), py::arg("m"),
             docs::kGaussianXPowerInit);
    EnvelopeClass<Custom>(m, "Custom", docs::kCustomClass)
        .def(py::init([](double start, double dt, const py::array& data) {
                 return std::static_pointer_cast<Custom>(MakeCustom(start, dt, data));
             }),
             py::arg("start"), py::arg("dt"), py::arg("data"), docs::kCustomInit);

    auto mix_exp = py::class_<MixExp, AbstractEnvelope, std::shared_ptr<MixExp>>(m, "MixExp");
    AddPickle<MixExp>(mix_exp);
    auto env_mix = py::class_<EnvMix, AbstractEnvelope, std::shared_ptr<EnvMix>>(m, "EnvMix");
    AddPickle<EnvMix>(env_mix);
    auto env_shift = py::class_<EnvShift, AbstractEnvelope, std::shared_ptr<EnvShift>>(m, "EnvShift");
    AddPickle<EnvShift>(env_shift);
    auto real_prod = py::class_<EnvProd<float>, AbstractEnvelope, std::shared_ptr<EnvProd<float>>>(m, "RealEnvProd");
    AddPickle<EnvProd<float>>(real_prod);
    auto complex_prod = py::class_<EnvProd<CFloat>, AbstractEnvelope, std::shared_ptr<EnvProd<CFloat>>>(
        m, "COMPLEXEnvProd");
    AddPickle<EnvProd<CFloat>>(complex_prod);
    auto env_sum = py::class_<EnvSum, AbstractEnvelope, std::shared_ptr<EnvSum>>(m, "EnvSum");
    AddPickle<EnvSum>(env_sum);

    m.def(
        "decode_envelope", [](const EnvPtr& env, WaveCache& wc) { return ToPython(DecodeEnvelope(Required(env), wc)); },
        py::arg("env"), py::arg("wc"), docs::kDecodeEnvelope0);
    m.def(
        "decode_envelope",
        [](const EnvPtr& env, WaveCache& wc, double start, double end) {
            return ToPython(DecodeEnvelope(Required(env), wc, start, end));
        },
        py::arg("env"), py::arg("wc"), py::arg("start"), py::arg("end"), docs::kDecodeEnvelope1);
    m.def(
        "serialization", [](const EnvPtr& env) { return py::bytes(Serialize(Required(env))); }, py::arg("env"),
        docs::kSerialization0);
    m.def(
        "deserialization", [](const py::bytes& serialized) { return Deserialize(serialized); }, py::arg("serialized"),
        docs::kDeserialization0);
    m.def(
        "align", [](const EnvPtr& env, const InArray<double>& dt) { return Align(Required(env), ToVector(dt)); },
        py::arg("env"), py::arg("dt"), docs::kAlign0);
    m.def(
        "align",
        [](const EnvPtr& env, const InArray<double>& dt, const InArray<float>& amp) {
            return Align(Required(env), ToVector(dt), ToVector(amp));
        },
        py::arg("env"), py::arg("dt"), py::arg("amp"), docs::kAlign1);
    m.def(
        "align",
        [](const EnvPtr& env, const InArray<double>& dt, const InArray<CFloat>& amp) {
            return Align(Required(env), ToVector(dt), ToVector(amp));
        },
        py::arg("env"), py::arg("dt"), py::arg("amp"), docs::kAlign2);
    m.def(
        "split",
        [](const EnvPtr& env, const InArray<double>& starts, const InArray<double>& ends) {
            return Split(Required(env), ToVector(starts), ToVector(ends));
        },
        py::arg("env"), py::arg("starts"), py::arg("ends"), docs::kSplit0);
    m.def(
        "mix",
        [](const EnvPtr& env, double df, double phase, bool dynamical) {
            return Mix(Required(env), df, phase, dynamical);
        },
        py::arg("env"), py::arg("df"), py::arg("phase"), py::arg("dynamical"), docs::kMix0);
}

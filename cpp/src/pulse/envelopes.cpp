#include "quspect_native/pulse/envelopes.hpp"

#include <cmath>

#include <algorithm>
#include <cstring>
#include <functional>
#include <sstream>
#include <stdexcept>

#include "cerf.h"
#include "quspect_native/pulse/sha256.hpp"

namespace quspect::native::pulse {

namespace {

constexpr double kPi = 3.141592653589793;
constexpr double kTwoPi = 6.283185307179586;
constexpr double kNegTwoPi = -6.283185307179586;
constexpr double kFwhmToSigma = 2.3548200450309493;  // sqrt(8 ln 2)
constexpr double kTwoSqrtLn2 = 1.6651092223153954;
constexpr double kSqrtLn2 = 0.8325546111576977;

long long Quantize(double value, double atol) {
    return std::llround(value / atol);
}

template <class T>
void Put(std::string& out, const T& value) {
    out.append(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <class T>
WavePtr<T> MakeWave(size_t n) {
    return std::make_shared<std::vector<T>>(n);
}

// Number of cached samples for an envelope of the given duration.
size_t SampleCount(double duration, double res) {
    return static_cast<size_t>(std::llround(std::floor(duration / res + 0.5)) + kEndPadding);
}

// Cheap downcast by type tag (dynamic_cast chains dominate Align() otherwise).
template <class D>
const D* As(const EnvPtr& p);

template <>
const EnvSum* As<EnvSum>(const EnvPtr& p) {
    return p->Type() == EnvType::kEnvSum ? static_cast<const EnvSum*>(p.get()) : nullptr;
}

template <>
const EnvShift* As<EnvShift>(const EnvPtr& p) {
    return p->Type() == EnvType::kEnvShift ? static_cast<const EnvShift*>(p.get()) : nullptr;
}

template <>
const EnvProd<float>* As<EnvProd<float>>(const EnvPtr& p) {
    return p->Type() == EnvType::kEnvProd && !p->IsComplexProd() ? static_cast<const EnvProd<float>*>(p.get())
                                                                 : nullptr;
}

template <>
const EnvProd<CFloat>* As<EnvProd<CFloat>>(const EnvPtr& p) {
    return p->Type() == EnvType::kEnvProd && p->IsComplexProd() ? static_cast<const EnvProd<CFloat>*>(p.get())
                                                                : nullptr;
}

template <>
const Envelope* As<Envelope>(const EnvPtr& p) {
    return static_cast<uint8_t>(p->Type()) <= static_cast<uint8_t>(EnvType::kCustom)
               ? static_cast<const Envelope*>(p.get())
               : nullptr;
}

}  // namespace

// ---------------- EnvFeature / WaveCache ----------------

std::string EnvFeature::ToString() const {
    std::ostringstream os;
    os << "[";
    for (long long v : values)
        os << v << " ";
    os << "]";
    return os.str();
}

size_t EnvFeature::Hash::operator()(const EnvFeature& f) const {
    size_t seed = f.values.size();
    for (long long v : f.values) {
        seed ^= std::hash<long long>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
}

WaveCache::WaveCache(double resolution)
    : resolution_(resolution), resolution_over_atol_(Quantize(resolution, kTimeAtol)) {
    if (resolution <= 0)
        throw std::invalid_argument("WaveCache resolution must be positive.");
    if (resolution < kTimeAtol)
        throw std::invalid_argument("WaveCache resolution is too small.");
}

template <>
NameMap<float>& WaveCache::Maps<float>() {
    return real_cache_;
}

template <>
NameMap<CFloat>& WaveCache::Maps<CFloat>() {
    return complex_cache_;
}

template <class T>
std::tuple<double, double, FeatureMap<T>*> WaveCache::GetCache(const std::string& name, double start) {
    long long start_over_atol = Quantize(start, kTimeAtol);
    long long q = start_over_atol / resolution_over_atol_;  // truncating, like the reference
    long long base = start_over_atol - q * resolution_over_atol_;
    auto& slot = Maps<T>()[name][base];
    if (!slot)
        slot = std::make_shared<FeatureMap<T>>();
    return {static_cast<double>(base) * kTimeAtol, resolution_ * static_cast<double>(q), slot.get()};
}

template std::tuple<double, double, FeatureMap<float>*> WaveCache::GetCache<float>(const std::string&, double);
template std::tuple<double, double, FeatureMap<CFloat>*> WaveCache::GetCache<CFloat>(const std::string&, double);

namespace {

template <class T>
void DumpCache(std::ostringstream& os, const NameMap<T>& cache) {
    for (const auto& [name, bases] : cache) {
        os << "  " << name << ":\n";
        for (const auto& [base, features] : bases) {
            os << "    " << base << ":\n";
            for (const auto& [feature, wave] : *features)
                os << "      " << feature.ToString() << "\n";
        }
    }
}

}  // namespace

std::string WaveCache::ToString() const {
    std::ostringstream os;
    os << "resolution: " << resolution_ << "\n\nreal_cache:\n";
    DumpCache(os, real_cache_);
    os << "\ncomplex_cache:\n";
    DumpCache(os, complex_cache_);
    return os.str();
}

// ---------------- LazyArray ----------------

template <class T>
LazyArray<T>& LazyArray<T>::operator*=(const LazyArray& other) {
    if (wave->size() != other.wave->size()) {
        throw std::runtime_error("LazyArray cannot be multiplied by other LazyArray with different sizes.");
    }
    amp = amp * other.amp;
    extras.push_back(other.wave);
    extras.insert(extras.end(), other.extras.begin(), other.extras.end());
    return *this;
}

template <class T>
WavePtr<T> LazyArray<T>::Eval() const {
    auto out = MakeWave<T>(wave->size());
    for (size_t k = 0; k < wave->size(); ++k) {
        T v = (*wave)[k] * amp;
        for (const auto& e : extras)
            v = v * (*e)[k];
        (*out)[k] = v;
    }
    return out;
}

template struct LazyArray<float>;
template struct LazyArray<CFloat>;

// ---------------- Envelope ----------------

Envelope::Envelope(double start, double end, float amp, bool is_complex, std::vector<long long> feature)
    : start_(start), end_(end), amp_(amp), complex_(is_complex), feature_{std::move(feature)} {
}

TimeFunc<float> Envelope::RealTimeFunc(WaveCache& wc, double dt) const {
    auto [base_start, shift_t, features] = wc.GetCache<float>(CacheName(), Start() + dt);
    auto it = features->find(feature_);
    if (it == features->end()) {
        double res = wc.resolution();
        auto wave = RealModel(base_start, SampleCount(Duration(), res), res);
        it = features->emplace(feature_, std::move(wave)).first;
    }
    return {shift_t, std::make_shared<LazyArray<float>>(amp_, it->second)};
}

TimeFunc<CFloat> Envelope::ComplexTimeFunc(WaveCache& wc, double dt) const {
    auto [base_start, shift_t, features] = wc.GetCache<CFloat>(CacheName(), Start() + dt);
    auto it = features->find(feature_);
    if (it == features->end()) {
        double res = wc.resolution();
        auto wave = ComplexModel(base_start, SampleCount(Duration(), res), res);
        it = features->emplace(feature_, std::move(wave)).first;
    }
    return {shift_t, std::make_shared<LazyArray<CFloat>>(CFloat(amp_, 0.0f), it->second)};
}

WavePtr<float> Envelope::RealModel(double, size_t, double) const {
    throw std::runtime_error("NOT implemented.");
}

WavePtr<CFloat> Envelope::ComplexModel(double base_start, size_t n, double res) const {
    auto real = RealModel(base_start, n, res);
    auto out = MakeWave<CFloat>(n);
    for (size_t k = 0; k < n; ++k)
        (*out)[k] = CFloat((*real)[k], 0.0f);
    return out;
}

// Shared by the DRAG pulses: rotate the cached wave by the carrier phase at shift_t.
static TimeFunc<CFloat> ApplyCarrier(TimeFunc<CFloat> tf, double df, double phase) {
    double ph = phase + (df * kNegTwoPi) * tf.t;
    CFloat rot(static_cast<float>(std::cos(ph)), static_cast<float>(std::sin(ph)));
    tf.lazy->amp = tf.lazy->amp * rot;
    return tf;
}

// Gaussian

Gaussian::Gaussian(double t0, double w, float amp)
    : Envelope(t0 - w * 3.0, w * 3.0 + t0, amp, false, {Quantize(w, kTimeAtol)})
    , t0_(t0)
    , w_(w)
    , sigma_(w / kFwhmToSigma) {
}

WavePtr<float> Gaussian::RealModel(double base_start, size_t n, double res) const {
    auto out = MakeWave<float>(n);
    double t0 = w_ * 3.0 + base_start;
    double denom = sigma_ * (sigma_ + sigma_);
    for (size_t k = 0; k < n; ++k) {
        double x = static_cast<double>(k) * res - t0;
        (*out)[k] = static_cast<float>(std::exp((x * -x) / denom));
    }
    return out;
}

void Gaussian::SerializeState(std::string& out) const {
    Put(out, t0_);
    Put(out, w_);
    Put(out, amp_);
}

EnvPtr Gaussian::Rebuild(const std::vector<EnvPtr>&) const {
    return std::make_shared<Gaussian>(t0_, w_, amp_);
}

// GaussianDrag

GaussianDrag::GaussianDrag(double t0, double w, float amp, double coef, double df, double phase)
    : Envelope(t0 - w * 3.0, w * 3.0 + t0, amp, true,
               {Quantize(w, kTimeAtol), Quantize(coef, kFloatAtol), Quantize(df, kFreqAtol)})
    , t0_(t0)
    , w_(w)
    , sigma_(w / kFwhmToSigma)
    , coef_(coef)
    , df_(df)
    , phase_(phase) {
}

WavePtr<CFloat> GaussianDrag::ComplexModel(double base_start, size_t n, double res) const {
    auto out = MakeWave<CFloat>(n);
    double t0 = w_ * 3.0 + base_start;
    double s2 = sigma_ * sigma_;
    for (size_t k = 0; k < n; ++k) {
        double t = static_cast<double>(k) * res;
        double ph = t * (df_ * kNegTwoPi);
        CFloat rot(static_cast<float>(std::cos(ph)), static_cast<float>(std::sin(ph)));
        double x = t - t0;
        double g = std::exp((x * x * -0.5) / s2);
        double re = g * ((x * (coef_ * -0.0)) / s2 + 1.0);
        double im = ((x * -coef_) / s2) * g;
        (*out)[k] = rot * CFloat(static_cast<float>(re), static_cast<float>(im));
    }
    return out;
}

TimeFunc<CFloat> GaussianDrag::ComplexTimeFunc(WaveCache& wc, double dt) const {
    return ApplyCarrier(Envelope::ComplexTimeFunc(wc, dt), df_, phase_);
}

void GaussianDrag::SerializeState(std::string& out) const {
    Put(out, t0_);
    Put(out, w_);
    Put(out, amp_);
    Put(out, coef_);
    Put(out, df_);
    Put(out, phase_);
}

EnvPtr GaussianDrag::Rebuild(const std::vector<EnvPtr>&) const {
    return std::make_shared<GaussianDrag>(t0_, w_, amp_, coef_, df_, phase_);
}

// CosineDrag

CosineDrag::CosineDrag(double t0, double w, float amp, double coef, double df, double phase)
    : Envelope(t0 - w * 0.5, w * 0.5 + t0, amp, true,
               {Quantize(w, kTimeAtol), Quantize(coef, kFloatAtol), Quantize(df, kFreqAtol)})
    , t0_(t0)
    , w_(w)
    , coef_(coef)
    , df_(df)
    , phase_(phase) {
}

WavePtr<CFloat> CosineDrag::ComplexModel(double base_start, size_t n, double res) const {
    auto out = MakeWave<CFloat>(n);
    double t0 = w_ * 0.5 + base_start;
    for (size_t k = 0; k < n; ++k) {
        double t = static_cast<double>(k) * res;
        double ph = t * (df_ * kNegTwoPi);
        CFloat rot(static_cast<float>(std::cos(ph)), static_cast<float>(std::sin(ph)));
        double x = t - t0;
        double a = (x * kTwoPi) / w_;
        double s = std::sin(a);
        double c = std::cos(a);
        double re = (c + 1.0) * 0.5 + s * (((coef_ * -0.0) * kPi) / w_);
        double im = s * ((coef_ * -kPi) / w_);
        CFloat v = rot * CFloat(static_cast<float>(re), static_cast<float>(im));
        bool inside = (x > w_ * -0.5) && (x < w_ * 0.5);
        (*out)[k] = v * CFloat(static_cast<float>(inside), 0.0f);
    }
    return out;
}

TimeFunc<CFloat> CosineDrag::ComplexTimeFunc(WaveCache& wc, double dt) const {
    return ApplyCarrier(Envelope::ComplexTimeFunc(wc, dt), df_, phase_);
}

void CosineDrag::SerializeState(std::string& out) const {
    Put(out, t0_);
    Put(out, w_);
    Put(out, amp_);
    Put(out, coef_);
    Put(out, df_);
    Put(out, phase_);
}

EnvPtr CosineDrag::Rebuild(const std::vector<EnvPtr>&) const {
    return std::make_shared<CosineDrag>(t0_, w_, amp_, coef_, df_, phase_);
}

// Triangle

Triangle::Triangle(double t0, double tlen, float amp, bool fall)
    : Envelope(t0, t0 + tlen, amp, false, {Quantize(tlen, kTimeAtol), fall ? 1 : 0})
    , t0_(t0)
    , tlen_(tlen)
    , fall_(fall) {
}

WavePtr<float> Triangle::RealModel(double base_start, size_t n, double res) const {
    auto out = MakeWave<float>(n);
    for (size_t k = 0; k < n; ++k) {
        double x = static_cast<double>(k) * res - base_start;
        double inside = static_cast<double>(x >= 0.0 && x < tlen_);
        double v = fall_ ? (1.0 - x / tlen_) * inside : (x / tlen_) * inside;
        (*out)[k] = static_cast<float>(v);
    }
    return out;
}

void Triangle::SerializeState(std::string& out) const {
    Put(out, t0_);
    Put(out, tlen_);
    Put(out, amp_);
    Put(out, static_cast<uint8_t>(fall_));
}

EnvPtr Triangle::Rebuild(const std::vector<EnvPtr>&) const {
    return std::make_shared<Triangle>(t0_, tlen_, amp_, fall_);
}

// Rect

Rect::Rect(double t0, double tlen, float amp)
    : Envelope(t0, t0 + tlen, amp, false, {Quantize(tlen, kTimeAtol)}), t0_(t0), tlen_(tlen) {
}

WavePtr<float> Rect::RealModel(double base_start, size_t n, double res) const {
    auto out = MakeWave<float>(n);
    for (size_t k = 0; k < n; ++k) {
        double x = static_cast<double>(k) * res - base_start;
        (*out)[k] = (x >= 0.0) ? static_cast<float>(x < tlen_) : 0.0f;
    }
    return out;
}

void Rect::SerializeState(std::string& out) const {
    Put(out, t0_);
    Put(out, tlen_);
    Put(out, amp_);
}

EnvPtr Rect::Rebuild(const std::vector<EnvPtr>&) const {
    return std::make_shared<Rect>(t0_, tlen_, amp_);
}

// Flattop

Flattop::Flattop(double t0, double tlen, double w_left, double w_right, float amp)
    : Envelope(t0 - w_left * 3.0, (t0 + tlen) + w_right * 3.0, amp, false,
               {Quantize(w_left, kTimeAtol), Quantize(w_right, kTimeAtol), Quantize(tlen, kTimeAtol)})
    , t0_(t0)
    , tlen_(tlen)
    , w_left_(w_left)
    , w_right_(w_right)
    , a_left_(kTwoSqrtLn2 / w_left)
    , a_right_(kTwoSqrtLn2 / w_right) {
}

WavePtr<float> Flattop::RealModel(double base_start, size_t n, double res) const {
    auto out = MakeWave<float>(n);
    double t0 = w_left_ * 3.0 + base_start;
    double t1 = t0 + tlen_;
    for (size_t k = 0; k < n; ++k) {
        double t = static_cast<double>(k) * res;
        double v = (std::erf(a_right_ * (t1 - t)) - std::erf(a_left_ * (t0 - t))) * 0.5;
        (*out)[k] = static_cast<float>(v);
    }
    return out;
}

void Flattop::SerializeState(std::string& out) const {
    Put(out, t0_);
    Put(out, tlen_);
    Put(out, w_left_);
    Put(out, w_right_);
    Put(out, amp_);
}

EnvPtr Flattop::Rebuild(const std::vector<EnvPtr>&) const {
    return std::make_shared<Flattop>(t0_, tlen_, w_left_, w_right_, amp_);
}

// RippleRect

RippleRect::RippleRect(double t0, double tlen, float amp, double w, double ripple0, double ripple1, double ripple2,
                       double ripple3)
    : Envelope(t0 - w * 3.0, (t0 + tlen) + w * 3.0, amp, false,
               {Quantize(ripple0, kTimeAtol), Quantize(ripple1, kTimeAtol), Quantize(ripple2, kTimeAtol),
                Quantize(ripple3, kTimeAtol), Quantize(w, kTimeAtol), Quantize(tlen, kTimeAtol)})
    , t0_(t0)
    , tlen_(tlen)
    , w_(w)
    , ripples_{ripple0, ripple1, ripple2, ripple3} {
}

WavePtr<float> RippleRect::RealModel(double base_start, size_t n, double res) const {
    using cd = std::complex<double>;
    auto out = MakeWave<float>(n);
    double tmin = w_ * 3.0 + base_start;
    double tmax = tlen_ + tmin;
    double tmid = (tmin + tmax) * 0.5;
    double sigma = ((w_ + 1e-10) * 0.5) / kSqrtLn2;  // NOT the sigma of a Gaussian
    double half_base_amp = ((1.0 - ripples_[1]) - ripples_[3]) * 0.5;
    // Real part of 1j * idx_r * pi * sigma / 2, kept for identical signed-zero handling.
    double shift_re = ((0.0 * sigma) * 0.5);
    for (size_t k = 0; k < n; ++k) {
        double t = static_cast<double>(k) * res;
        double a = (t - tmin) / sigma;
        double b = (t - tmax) / sigma;
        double v = half_base_amp * (std::erf(a) - std::erf(b));
        double x = t - tmid;
        for (int idx = 0; idx < 4; ++idx) {
            double idx_pi = (idx < 2) ? kPi : kTwoPi;
            double q = ((sigma * idx_pi) * 0.5) / tlen_;
            double damp = std::exp(q * -q);
            cd rot = std::exp(cd((x * 0.0) / tlen_, (x * idx_pi) / tlen_));
            cd za(a + shift_re / tlen_, q);
            cd zb(b + shift_re / tlen_, q);
            cd ea = cerf(za);
            cd eb = cerf(zb);
            cd prod = rot * cd(ea.real() - eb.real(), ea.imag() - eb.imag());
            double part = (idx % 2 == 0) ? prod.imag() : prod.real();
            v = v + ((ripples_[idx] * 0.5) * damp) * part;
        }
        (*out)[k] = static_cast<float>(v);
    }
    return out;
}

void RippleRect::SerializeState(std::string& out) const {
    Put(out, t0_);
    Put(out, tlen_);
    Put(out, amp_);
    Put(out, w_);
    for (double r : ripples_)
        Put(out, r);
}

EnvPtr RippleRect::Rebuild(const std::vector<EnvPtr>&) const {
    return std::make_shared<RippleRect>(t0_, tlen_, amp_, w_, ripples_[0], ripples_[1], ripples_[2], ripples_[3]);
}

// PowerSineCosine

PowerSineCosine::PowerSineCosine(double t0, double tlen, float amp, int m, int n, double k, double phase)
    : Envelope(t0, t0 + tlen, amp, false,
               {Quantize(tlen, kTimeAtol), m, n, Quantize(k, kFloatAtol), Quantize(phase, kFloatAtol)})
    , t0_(t0)
    , tlen_(tlen)
    , m_(m)
    , n_(n)
    , k_(k)
    , phase_(phase) {
}

WavePtr<float> PowerSineCosine::RealModel(double base_start, size_t n, double res) const {
    auto out = MakeWave<float>(n);
    double k_pi = k_ * kPi;
    double m = static_cast<double>(m_);
    double nn = static_cast<double>(n_);
    for (size_t k = 0; k < n; ++k) {
        double x = phase_ + (k_pi * (static_cast<double>(k) * res - base_start)) / tlen_;
        double v = std::pow(std::sin(x), m) * std::pow(std::cos(x), nn);
        (*out)[k] = static_cast<float>(v);
    }
    return out;
}

void PowerSineCosine::SerializeState(std::string& out) const {
    Put(out, t0_);
    Put(out, tlen_);
    Put(out, amp_);
    Put(out, static_cast<int32_t>(m_));
    Put(out, static_cast<int32_t>(n_));
    Put(out, k_);
    Put(out, phase_);
}

EnvPtr PowerSineCosine::Rebuild(const std::vector<EnvPtr>&) const {
    return std::make_shared<PowerSineCosine>(t0_, tlen_, amp_, m_, n_, k_, phase_);
}

// GaussianXPower

GaussianXPower::GaussianXPower(double t0, double w, float amp, int m)
    : Envelope(t0 - w * 3.0, w * 3.0 + t0, amp, false, {Quantize(w, kTimeAtol), m})
    , t0_(t0)
    , w_(w)
    , sigma_(w / kFwhmToSigma)
    , m_(m) {
}

WavePtr<float> GaussianXPower::RealModel(double base_start, size_t n, double res) const {
    auto out = MakeWave<float>(n);
    double t0 = w_ * 3.0 + base_start;
    double denom = sigma_ * (sigma_ + sigma_);
    double m = static_cast<double>(m_);
    for (size_t k = 0; k < n; ++k) {
        double x = static_cast<double>(k) * res - t0;
        double g = std::exp((x * -x) / denom);
        (*out)[k] = static_cast<float>(std::pow(x, m) * g);
    }
    return out;
}

void GaussianXPower::SerializeState(std::string& out) const {
    Put(out, t0_);
    Put(out, w_);
    Put(out, amp_);
    Put(out, static_cast<int32_t>(m_));
}

EnvPtr GaussianXPower::Rebuild(const std::vector<EnvPtr>&) const {
    return std::make_shared<GaussianXPower>(t0_, w_, amp_, m_);
}

// Custom

Custom::Custom(double start, double dt, WavePtr<float> data)
    : Envelope(start, static_cast<double>(data->size()) * dt + start, 1.0f, false, {})
    , dt_(dt)
    , real_data_(std::move(data)) {
    InitId();
}

Custom::Custom(double start, double dt, WavePtr<CFloat> data)
    : Envelope(start, static_cast<double>(data->size()) * dt + start, 1.0f, true, {})
    , dt_(dt)
    , complex_data_(std::move(data)) {
    InitId();
}

void Custom::InitId() {
    if (!(dt_ > 0))
        throw std::runtime_error("Custom envelope dt must be positive.");
    SHA256 h;
    h.Update(&dt_, sizeof(dt_));
    uint8_t is_complex = complex_ ? 1 : 0;
    h.Update(&is_complex, 1);
    uint64_t size = complex_ ? complex_data_->size() : real_data_->size();
    h.Update(&size, sizeof(size));
    if (complex_) {
        h.Update(complex_data_->data(), complex_data_->size() * sizeof(CFloat));
    } else {
        h.Update(real_data_->data(), real_data_->size() * sizeof(float));
    }
    h.Final(id_);
    feature_.values.resize(4);
    std::memcpy(feature_.values.data(), id_, sizeof(id_));
}

std::string Custom::ToString() const {
    static const char* hex = "0123456789abcdef";
    std::string s = "Custom_";
    for (unsigned char c : id_) {
        s += hex[c >> 4];
        s += hex[c & 0xf];
    }
    return s;
}

WavePtr<float> Custom::RealModel(double base_start, size_t n, double res) const {
    if (complex_) {
        throw std::runtime_error("Custom envelope with complex data does not support real_model.");
    }
    auto out = MakeWave<float>(n);
    const auto& data = *real_data_;
    size_t last = data.size() - 1;
    double last_d = static_cast<double>(last);
    for (size_t k = 0; k < n; ++k) {
        double u = (static_cast<double>(k) * res - base_start) / dt_;
        if (u < 0.0 || u > last_d)
            continue;
        size_t i = static_cast<size_t>(std::floor(u));
        if (i < last) {
            double frac = u - static_cast<double>(i);
            (*out)[k] = data[i] * static_cast<float>(1.0 - frac) + data[i + 1] * static_cast<float>(frac);
        } else {
            (*out)[k] = data[last];
        }
    }
    return out;
}

WavePtr<CFloat> Custom::ComplexModel(double base_start, size_t n, double res) const {
    if (!complex_)
        return Envelope::ComplexModel(base_start, n, res);
    auto out = MakeWave<CFloat>(n);
    const auto& data = *complex_data_;
    size_t last = data.size() - 1;
    double last_d = static_cast<double>(last);
    for (size_t k = 0; k < n; ++k) {
        double u = (static_cast<double>(k) * res - base_start) / dt_;
        if (u < 0.0 || u > last_d)
            continue;
        size_t i = static_cast<size_t>(std::floor(u));
        if (i < last) {
            double frac = u - static_cast<double>(i);
            CFloat lo = data[i] * CFloat(static_cast<float>(1.0 - frac), 0.0f);
            CFloat hi = data[i + 1] * CFloat(static_cast<float>(frac), 0.0f);
            (*out)[k] = lo + hi;
        } else {
            (*out)[k] = data[last];
        }
    }
    return out;
}

void Custom::SerializeState(std::string& out) const {
    Put(out, start_);
    Put(out, dt_);
    Put(out, static_cast<uint8_t>(complex_));
    if (complex_) {
        Put(out, static_cast<uint64_t>(complex_data_->size()));
        out.append(reinterpret_cast<const char*>(complex_data_->data()), complex_data_->size() * sizeof(CFloat));
    } else {
        Put(out, static_cast<uint64_t>(real_data_->size()));
        out.append(reinterpret_cast<const char*>(real_data_->data()), real_data_->size() * sizeof(float));
    }
}

EnvPtr Custom::Rebuild(const std::vector<EnvPtr>&) const {
    if (complex_)
        return std::make_shared<Custom>(start_, dt_, complex_data_);
    return std::make_shared<Custom>(start_, dt_, real_data_);
}

// ---------------- MixExp / EnvMix ----------------

MixExp::MixExp(double start, double end, double df, double phase, bool dynamical)
    : start_(start), end_(end), df_(df), phase_(phase), dynamical_(dynamical) {
    feature_.values = {Quantize(df_, kFreqAtol), Quantize(Duration(), kMixDurationAtol)};
}

TimeFunc<float> MixExp::RealTimeFunc(WaveCache&, double) const {
    throw std::runtime_error("COMPLEX MixExp should not call real_time_func!");
}

TimeFunc<CFloat> MixExp::ComplexTimeFunc(WaveCache& wc, double dt) const {
    double res = wc.resolution();
    auto features = std::get<2>(wc.GetCache<CFloat>("MixExp", 0.0));
    double shift_t = res * static_cast<double>(Quantize(Start() + dt, kTimeAtol) / wc.resolution_over_atol());
    double w = df_ * kTwoPi;
    double ph = dynamical_ ? phase_ - shift_t * w : phase_ - (shift_t - dt) * w;
    CFloat rot(static_cast<float>(std::cos(ph)), static_cast<float>(std::sin(ph)));
    auto it = features->find(feature_);
    if (it == features->end()) {
        size_t n = SampleCount(Duration(), res);
        auto wave = MakeWave<CFloat>(n);
        for (size_t k = 0; k < n; ++k) {
            double p = (res * static_cast<double>(k)) * (df_ * kNegTwoPi);
            (*wave)[k] = CFloat(static_cast<float>(std::cos(p)), static_cast<float>(std::sin(p)));
        }
        it = features->emplace(feature_, std::move(wave)).first;
    }
    return {shift_t, std::make_shared<LazyArray<CFloat>>(rot, it->second)};
}

void MixExp::SerializeState(std::string& out) const {
    Put(out, start_);
    Put(out, end_);
    Put(out, df_);
    Put(out, phase_);
    Put(out, static_cast<uint8_t>(dynamical_));
}

EnvPtr MixExp::Rebuild(const std::vector<EnvPtr>&) const {
    return std::make_shared<MixExp>(start_, end_, df_, phase_, dynamical_);
}

EnvMix::EnvMix(EnvPtr env, double df, double phase, bool dynamical) : env_(std::move(env)) {
    carrier_ = std::make_shared<MixExp>(env_->Start(), env_->End(), df, phase, dynamical);
}

TimeFunc<float> EnvMix::RealTimeFunc(WaveCache&, double) const {
    throw std::runtime_error("COMPLEX EnvMix should not call real_time_func!");
}

TimeFunc<CFloat> EnvMix::ComplexTimeFunc(WaveCache& wc, double dt) const {
    auto tf = env_->ComplexTimeFunc(wc, dt);
    auto carrier = carrier_->ComplexTimeFunc(wc, dt);
    *tf.lazy *= *carrier.lazy;
    return tf;
}

EnvPtr EnvMix::Rebuild(const std::vector<EnvPtr>& children) const {
    auto exp = std::dynamic_pointer_cast<MixExp>(children.at(1));
    if (!exp)
        throw std::runtime_error("EnvMix expects a MixExp carrier.");
    return std::make_shared<EnvMix>(children.at(0), exp);
}

// ---------------- EnvShift / EnvProd ----------------

void EnvShift::SerializeState(std::string& out) const {
    Put(out, dt_);
}

EnvPtr EnvShift::Rebuild(const std::vector<EnvPtr>& children) const {
    return std::make_shared<EnvShift>(children.at(0), dt_);
}

template <>
bool EnvProd<float>::IsComplex() const {
    return env_->IsComplex();
}

template <>
bool EnvProd<CFloat>::IsComplex() const {
    return true;
}

template <>
TimeFunc<float> EnvProd<float>::RealTimeFunc(WaveCache& wc, double dt) const {
    auto tf = env_->RealTimeFunc(wc, dt);
    tf.lazy->amp = c_ * tf.lazy->amp;
    return tf;
}

template <>
TimeFunc<CFloat> EnvProd<float>::ComplexTimeFunc(WaveCache& wc, double dt) const {
    auto tf = env_->ComplexTimeFunc(wc, dt);
    CFloat a = tf.lazy->amp;
    tf.lazy->amp = CFloat(a.real() * c_, a.imag() * c_);
    return tf;
}

template <>
TimeFunc<float> EnvProd<CFloat>::RealTimeFunc(WaveCache&, double) const {
    throw std::runtime_error("COMPLEX EnvProd should not call real_time_func!");
}

template <>
TimeFunc<CFloat> EnvProd<CFloat>::ComplexTimeFunc(WaveCache& wc, double dt) const {
    auto tf = env_->ComplexTimeFunc(wc, dt);
    tf.lazy->amp = tf.lazy->amp * c_;
    return tf;
}

template <>
void EnvProd<float>::SerializeState(std::string& out) const {
    Put(out, static_cast<uint8_t>(0));
    Put(out, c_);
}

template <>
void EnvProd<CFloat>::SerializeState(std::string& out) const {
    Put(out, static_cast<uint8_t>(1));
    Put(out, c_.real());
    Put(out, c_.imag());
}

template <class T>
EnvPtr EnvProd<T>::Rebuild(const std::vector<EnvPtr>& children) const {
    return std::make_shared<EnvProd<T>>(children.at(0), c_);
}

template class EnvProd<float>;
template class EnvProd<CFloat>;

// ---------------- EnvSum ----------------

static std::runtime_error EmptySumError() {
    return std::runtime_error("EnvSum is empty.");
}

double EnvSum::Start() const {
    if (items_.empty())
        throw EmptySumError();
    double s = items_[0]->Start();
    for (size_t i = 1; i < items_.size(); ++i)
        s = std::min(s, items_[i]->Start());
    return s;
}

double EnvSum::End() const {
    if (items_.empty())
        throw EmptySumError();
    double e = items_[0]->End();
    for (size_t i = 1; i < items_.size(); ++i)
        e = std::max(e, items_[i]->End());
    return e;
}

bool EnvSum::IsComplex() const {
    return std::any_of(items_.begin(), items_.end(), [](const EnvPtr& e) { return e->IsComplex(); });
}

template <class T>
static TimeFunc<T> ItemTimeFunc(const EnvPtr& item, WaveCache& wc, double dt);

template <>
TimeFunc<float> ItemTimeFunc<float>(const EnvPtr& item, WaveCache& wc, double dt) {
    return item->RealTimeFunc(wc, dt);
}

template <>
TimeFunc<CFloat> ItemTimeFunc<CFloat>(const EnvPtr& item, WaveCache& wc, double dt) {
    return item->ComplexTimeFunc(wc, dt);
}

template <class T>
TimeFunc<T> EnvSum::SumTimeFunc(WaveCache& wc, double dt) const {
    if (items_.empty())
        throw EmptySumError();
    double res = wc.resolution();
    size_t count = items_.size();
    std::vector<TimeFunc<T>> parts;
    parts.reserve(count);
    std::vector<double> starts(count), ends(count);
    for (size_t i = 0; i < count; ++i) {
        parts.push_back(ItemTimeFunc<T>(items_[i], wc, dt));
        starts[i] = parts[i].t;
        ends[i] = parts[i].t + res * static_cast<double>(parts[i].lazy->wave->size());
    }
    double start = *std::min_element(starts.begin(), starts.end());
    double end = *std::max_element(ends.begin(), ends.end());
    auto out = MakeWave<T>(static_cast<size_t>(std::llround((end - start) / res)));
    T* dst = out->data();
    size_t out_size = out->size();
    for (size_t i = 0; i < count; ++i) {
        const LazyArray<T>& lazy = *parts[i].lazy;
        const std::vector<T>& wave = *lazy.wave;
        if (wave.empty())
            continue;
        long long offset = std::llround((starts[i] - start) / res);
        size_t len = wave.size();
        if (offset < 0 || static_cast<size_t>(offset) + len > out_size) {
            throw std::runtime_error("EnvSum item falls outside the summed range.");
        }
        T* row = dst + offset;
        const T amp = lazy.amp;
        if (lazy.extras.empty()) {
            for (size_t k = 0; k < len; ++k)
                row[k] = wave[k] * amp + row[k];
        } else {
            for (size_t k = 0; k < len; ++k) {
                T v = wave[k] * amp;
                for (const auto& e : lazy.extras)
                    v = v * (*e)[k];
                row[k] = v + row[k];
            }
        }
    }
    return {start, std::make_shared<LazyArray<T>>(T(1), out)};
}

TimeFunc<float> EnvSum::RealTimeFunc(WaveCache& wc, double dt) const {
    return SumTimeFunc<float>(wc, dt);
}

TimeFunc<CFloat> EnvSum::ComplexTimeFunc(WaveCache& wc, double dt) const {
    return SumTimeFunc<CFloat>(wc, dt);
}

std::string EnvSum::ToString() const {
    std::string s = "EnvSum(";
    for (size_t i = 0; i < items_.size(); ++i) {
        if (i)
            s += ", ";
        s += items_[i]->ToString();
    }
    return s + ")";
}

void EnvSum::SerializeState(std::string& out) const {
    Put(out, static_cast<uint64_t>(items_.size()));
}

EnvPtr EnvSum::Rebuild(const std::vector<EnvPtr>& children) const {
    return std::make_shared<EnvSum>(children);
}

void EnvSum::Append(const EnvPtr& env) {
    if (auto sum = As<EnvSum>(env)) {
        // Copy first: env may be this very sum.
        std::vector<EnvPtr> items = sum->items_;
        items_.insert(items_.end(), items.begin(), items.end());
    } else {
        items_.push_back(env);
    }
}

// ---------------- operations ----------------

EnvPtr Add(const EnvPtr& a, const EnvPtr& b) {
    auto sum = std::make_shared<EnvSum>();
    sum->Append(a);
    sum->Append(b);
    return sum;
}

EnvPtr Add(const EnvPtr& a, double c) {
    if (c == 0)
        return a;
    throw std::invalid_argument("Cannot add a non-zero constant to an envelope");
}

EnvPtr InplaceAdd(const EnvPtr& a, const EnvPtr& b) {
    if (auto sum = std::dynamic_pointer_cast<EnvSum>(a)) {
        sum->Append(b);
        return a;
    }
    return Add(a, b);
}

template <class F>
static EnvPtr MapSum(const EnvSum& sum, F f) {
    std::vector<EnvPtr> items;
    items.reserve(sum.items().size());
    for (const auto& item : sum.items())
        items.push_back(f(item));
    return std::make_shared<EnvSum>(std::move(items));
}

EnvPtr Multiply(const EnvPtr& a, float c) {
    if (auto sum = As<EnvSum>(a))
        return MapSum(*sum, [c](const EnvPtr& e) { return Multiply(e, c); });
    if (auto p = As<EnvProd<float>>(a))
        return std::make_shared<EnvProd<float>>(p->env(), p->constant() * c);
    if (auto p = As<EnvProd<CFloat>>(a)) {
        CFloat z = p->constant();
        return std::make_shared<EnvProd<CFloat>>(p->env(), CFloat(z.real() * c, z.imag() * c));
    }
    return std::make_shared<EnvProd<float>>(a, c);
}

EnvPtr Multiply(const EnvPtr& a, CFloat c) {
    if (auto sum = As<EnvSum>(a))
        return MapSum(*sum, [c](const EnvPtr& e) { return Multiply(e, c); });
    if (auto p = As<EnvProd<float>>(a)) {
        float r = p->constant();
        return std::make_shared<EnvProd<CFloat>>(p->env(), CFloat(c.real() * r, c.imag() * r));
    }
    if (auto p = As<EnvProd<CFloat>>(a))
        return std::make_shared<EnvProd<CFloat>>(p->env(), p->constant() * c);
    return std::make_shared<EnvProd<CFloat>>(a, c);
}

EnvPtr Shift(const EnvPtr& a, double dt) {
    if (auto sum = As<EnvSum>(a))
        return MapSum(*sum, [dt](const EnvPtr& e) { return Shift(e, dt); });
    if (auto s = As<EnvShift>(a))
        return std::make_shared<EnvShift>(s->env(), s->dt() + dt);
    if (auto p = As<EnvProd<float>>(a))
        return std::make_shared<EnvProd<float>>(Shift(p->env(), dt), p->constant());
    if (auto p = As<EnvProd<CFloat>>(a))
        return std::make_shared<EnvProd<CFloat>>(Shift(p->env(), dt), p->constant());
    return std::make_shared<EnvShift>(a, dt);
}

EnvPtr Mix(const EnvPtr& env, double df, double phase, bool dynamical) {
    if (auto sum = As<EnvSum>(env)) {
        return MapSum(*sum, [&](const EnvPtr& e) { return Mix(e, df, phase, dynamical); });
    }
    if (auto p = As<EnvProd<float>>(env)) {
        return std::make_shared<EnvProd<float>>(Mix(p->env(), df, phase, dynamical), p->constant());
    }
    if (auto p = As<EnvProd<CFloat>>(env)) {
        return std::make_shared<EnvProd<CFloat>>(Mix(p->env(), df, phase, dynamical), p->constant());
    }
    return std::make_shared<EnvMix>(env, df, phase, dynamical);
}

template <class F>
static EnvPtr AlignImpl(const EnvPtr& env, size_t count, F make) {
    std::vector<EnvPtr> items;
    if (auto sum = As<EnvSum>(env)) {
        items.reserve(sum->items().size() * count);
        for (const auto& item : sum->items()) {
            for (size_t i = 0; i < count; ++i)
                items.push_back(make(item, i));
        }
    } else {
        items.reserve(count);
        for (size_t i = 0; i < count; ++i)
            items.push_back(make(env, i));
    }
    return std::make_shared<EnvSum>(std::move(items));
}

EnvPtr Align(const EnvPtr& env, const std::vector<double>& dt) {
    return AlignImpl(env, dt.size(), [&](const EnvPtr& e, size_t i) { return Shift(e, dt[i]); });
}

template <class A>
static EnvPtr AlignAmp(const EnvPtr& env, const std::vector<double>& dt, const std::vector<A>& amp) {
    if (dt.size() != amp.size())
        throw std::runtime_error("dt and amp must have the same size.");
    return AlignImpl(env, dt.size(), [&](const EnvPtr& e, size_t i) { return Multiply(Shift(e, dt[i]), amp[i]); });
}

EnvPtr Align(const EnvPtr& env, const std::vector<double>& dt, const std::vector<float>& amp) {
    return AlignAmp(env, dt, amp);
}

EnvPtr Align(const EnvPtr& env, const std::vector<double>& dt, const std::vector<CFloat>& amp) {
    return AlignAmp(env, dt, amp);
}

std::vector<EnvPtr> Split(const EnvPtr& env, const std::vector<double>& starts, const std::vector<double>& ends) {
    if (starts.size() != ends.size()) {
        throw std::runtime_error("starts and ends must have the same size.");
    }
    if (starts.size() == 1) {
        if (env->Start() < starts[0] || ends[0] < env->End()) {
            throw std::runtime_error("Block can not cover the envelope!");
        }
        return {env};
    }
    std::vector<std::shared_ptr<EnvSum>> blocks;
    for (size_t j = 0; j < starts.size(); ++j)
        blocks.push_back(std::make_shared<EnvSum>());
    std::vector<EnvPtr> items;
    if (auto sum = As<EnvSum>(env))
        items = sum->items();
    else
        items.push_back(env);
    for (const auto& item : items) {
        bool found = false;
        double s = item->Start();
        double e = item->End();
        for (size_t j = 0; j < starts.size(); ++j) {
            if (s >= starts[j] && e <= ends[j]) {
                blocks[j]->Append(item);
                found = true;
            } else if ((s < starts[j] && starts[j] < e) || (s < ends[j] && ends[j] < e)) {
                throw std::runtime_error("Block can not cover the envelope! Block edge falls in env duration.");
            }
        }
        if (!found)
            throw std::runtime_error("There is env not in any block!");
    }
    bool is_complex = env->IsComplex();
    std::vector<EnvPtr> result;
    for (size_t j = 0; j < blocks.size(); ++j) {
        if (blocks[j]->items().empty())
            blocks[j]->Append(std::make_shared<Rect>(starts[j], 0.0, 0.0f));
        if (is_complex && !blocks[j]->IsComplex())
            result.push_back(Multiply(blocks[j], CFloat(1.0f, 0.0f)));
        else
            result.push_back(blocks[j]);
    }
    return result;
}

// Like the reference: shared leaf envelopes stay shared in the copy, while composite nodes
// (and the MixExp carrier of each EnvMix) are copied once per reference.
static EnvPtr CopyHelper(const EnvPtr& env, std::unordered_map<const AbstractEnvelope*, EnvPtr>& copied) {
    bool leaf = As<Envelope>(env) != nullptr;
    if (leaf) {
        auto it = copied.find(env.get());
        if (it != copied.end())
            return it->second;
    }
    std::vector<EnvPtr> children;
    for (const auto& child : env->Children())
        children.push_back(CopyHelper(child, copied));
    EnvPtr result = env->Rebuild(children);
    if (leaf)
        copied.emplace(env.get(), result);
    return result;
}

EnvPtr Copy(const EnvPtr& env) {
    std::unordered_map<const AbstractEnvelope*, EnvPtr> copied;
    return CopyHelper(env, copied);
}

// ---------------- serialization ----------------

namespace {

struct SerialNode {
    std::string state;
    std::vector<const AbstractEnvelope*> children;
};

void SerializationHelper(const EnvPtr& env, std::vector<SerialNode>& nodes,
                         std::unordered_map<const AbstractEnvelope*, uint32_t>& index) {
    if (index.count(env.get()))
        return;
    index.emplace(env.get(), static_cast<uint32_t>(nodes.size()));
    SerialNode node;
    node.state.push_back(static_cast<char>(env->Type()));
    env->SerializeState(node.state);
    auto children = env->Children();
    for (const auto& child : children)
        node.children.push_back(child.get());
    nodes.push_back(std::move(node));
    for (const auto& child : children)
        SerializationHelper(child, nodes, index);
}

class Reader {
 public:
    explicit Reader(const std::string& data) : data_(data) {
    }

    template <class T>
    T Get() {
        T value;
        Need(sizeof(T));
        std::memcpy(&value, data_.data() + pos_, sizeof(T));
        pos_ += sizeof(T);
        return value;
    }

    template <class T>
    WavePtr<T> GetArray(uint64_t n) {
        if (n > (data_.size() - pos_) / sizeof(T))
            Truncated();
        auto out = MakeWave<T>(n);
        std::memcpy(out->data(), data_.data() + pos_, n * sizeof(T));
        pos_ += n * sizeof(T);
        return out;
    }

 private:
    void Need(size_t n) const {
        if (data_.size() - pos_ < n)
            Truncated();
    }
    [[noreturn]] static void Truncated() {
        throw std::runtime_error("Serialized data is truncated.");
    }

    const std::string& data_;
    size_t pos_ = 0;
};

struct ParsedNode {
    EnvType type;
    EnvPtr leaf;  // set for nodes without children
    double dt = 0;
    bool prod_complex = false;
    float prod_real = 0;
    CFloat prod_cplx;
    uint64_t child_count = 0;
    std::vector<uint32_t> children;
};

ParsedNode ParseNode(Reader& r) {
    ParsedNode n;
    uint8_t tag = r.Get<uint8_t>();
    if (tag > static_cast<uint8_t>(EnvType::kEnvSum))
        throw std::runtime_error("Unknown envelope type.");
    n.type = static_cast<EnvType>(tag);
    switch (n.type) {
        case EnvType::kGaussian: {
            double t0 = r.Get<double>(), w = r.Get<double>();
            float amp = r.Get<float>();
            n.leaf = std::make_shared<Gaussian>(t0, w, amp);
            break;
        }
        case EnvType::kGaussianDrag:
        case EnvType::kCosineDrag: {
            double t0 = r.Get<double>(), w = r.Get<double>();
            float amp = r.Get<float>();
            double coef = r.Get<double>(), df = r.Get<double>(), phase = r.Get<double>();
            if (n.type == EnvType::kGaussianDrag) {
                n.leaf = std::make_shared<GaussianDrag>(t0, w, amp, coef, df, phase);
            } else {
                n.leaf = std::make_shared<CosineDrag>(t0, w, amp, coef, df, phase);
            }
            break;
        }
        case EnvType::kTriangle: {
            double t0 = r.Get<double>(), tlen = r.Get<double>();
            float amp = r.Get<float>();
            bool fall = r.Get<uint8_t>() != 0;
            n.leaf = std::make_shared<Triangle>(t0, tlen, amp, fall);
            break;
        }
        case EnvType::kRect: {
            double t0 = r.Get<double>(), tlen = r.Get<double>();
            float amp = r.Get<float>();
            n.leaf = std::make_shared<Rect>(t0, tlen, amp);
            break;
        }
        case EnvType::kFlattop: {
            double t0 = r.Get<double>(), tlen = r.Get<double>();
            double wl = r.Get<double>(), wr = r.Get<double>();
            float amp = r.Get<float>();
            n.leaf = std::make_shared<Flattop>(t0, tlen, wl, wr, amp);
            break;
        }
        case EnvType::kRippleRect: {
            double t0 = r.Get<double>(), tlen = r.Get<double>();
            float amp = r.Get<float>();
            double w = r.Get<double>();
            double r0 = r.Get<double>(), r1 = r.Get<double>(), r2 = r.Get<double>(), r3 = r.Get<double>();
            n.leaf = std::make_shared<RippleRect>(t0, tlen, amp, w, r0, r1, r2, r3);
            break;
        }
        case EnvType::kPowerSineCosine: {
            double t0 = r.Get<double>(), tlen = r.Get<double>();
            float amp = r.Get<float>();
            int m = r.Get<int32_t>(), nn = r.Get<int32_t>();
            double k = r.Get<double>(), phase = r.Get<double>();
            n.leaf = std::make_shared<PowerSineCosine>(t0, tlen, amp, m, nn, k, phase);
            break;
        }
        case EnvType::kGaussianXPower: {
            double t0 = r.Get<double>(), w = r.Get<double>();
            float amp = r.Get<float>();
            int m = r.Get<int32_t>();
            n.leaf = std::make_shared<GaussianXPower>(t0, w, amp, m);
            break;
        }
        case EnvType::kCustom: {
            double start = r.Get<double>(), dt = r.Get<double>();
            bool is_complex = r.Get<uint8_t>() != 0;
            uint64_t size = r.Get<uint64_t>();
            if (is_complex) {
                n.leaf = std::make_shared<Custom>(start, dt, r.GetArray<CFloat>(size));
            } else {
                n.leaf = std::make_shared<Custom>(start, dt, r.GetArray<float>(size));
            }
            break;
        }
        case EnvType::kMixExp: {
            double start = r.Get<double>(), end = r.Get<double>();
            double df = r.Get<double>(), phase = r.Get<double>();
            bool dynamical = r.Get<uint8_t>() != 0;
            n.leaf = std::make_shared<MixExp>(start, end, df, phase, dynamical);
            break;
        }
        case EnvType::kEnvMix:
            n.child_count = 2;
            break;
        case EnvType::kEnvShift:
            n.dt = r.Get<double>();
            n.child_count = 1;
            break;
        case EnvType::kEnvProd:
            n.prod_complex = r.Get<uint8_t>() != 0;
            if (n.prod_complex) {
                float re = r.Get<float>(), im = r.Get<float>();
                n.prod_cplx = CFloat(re, im);
            } else {
                n.prod_real = r.Get<float>();
            }
            n.child_count = 1;
            break;
        case EnvType::kEnvSum:
            n.child_count = r.Get<uint64_t>();
            break;
    }
    return n;
}

EnvPtr BuildNode(std::vector<ParsedNode>& nodes, std::vector<EnvPtr>& built, std::vector<char>& visiting, uint32_t i) {
    if (i >= nodes.size())
        throw std::runtime_error("Serialized node index out of range.");
    if (built[i])
        return built[i];
    if (visiting[i])
        throw std::runtime_error("Serialized envelope graph contains a cycle.");
    visiting[i] = 1;
    ParsedNode& n = nodes[i];
    std::vector<EnvPtr> children;
    for (uint32_t c : n.children)
        children.push_back(BuildNode(nodes, built, visiting, c));
    EnvPtr result;
    switch (n.type) {
        case EnvType::kEnvMix: {
            auto exp = std::dynamic_pointer_cast<MixExp>(children[1]);
            if (!exp)
                throw std::runtime_error("EnvMix expects a MixExp carrier.");
            result = std::make_shared<EnvMix>(children[0], exp);
            break;
        }
        case EnvType::kEnvShift:
            result = std::make_shared<EnvShift>(children[0], n.dt);
            break;
        case EnvType::kEnvProd:
            if (n.prod_complex)
                result = std::make_shared<EnvProd<CFloat>>(children[0], n.prod_cplx);
            else
                result = std::make_shared<EnvProd<float>>(children[0], n.prod_real);
            break;
        case EnvType::kEnvSum:
            result = std::make_shared<EnvSum>(std::move(children));
            break;
        default:
            result = n.leaf;
    }
    visiting[i] = 0;
    built[i] = result;
    return result;
}

}  // namespace

std::string Serialize(const EnvPtr& env) {
    std::vector<SerialNode> nodes;
    std::unordered_map<const AbstractEnvelope*, uint32_t> index;
    SerializationHelper(env, nodes, index);
    std::string out;
    Put(out, static_cast<uint64_t>(nodes.size()));
    for (const auto& node : nodes)
        out += node.state;
    for (const auto& node : nodes) {
        for (const auto* child : node.children)
            Put(out, index.at(child));
    }
    return out;
}

EnvPtr Deserialize(const std::string& data) {
    Reader r(data);
    uint64_t count = r.Get<uint64_t>();
    if (count == 0)
        throw std::runtime_error("Serialized data contains no envelope.");
    std::vector<ParsedNode> nodes;
    for (uint64_t i = 0; i < count; ++i)
        nodes.push_back(ParseNode(r));
    for (auto& n : nodes) {
        for (uint64_t c = 0; c < n.child_count; ++c)
            n.children.push_back(r.Get<uint32_t>());
    }
    std::vector<EnvPtr> built(nodes.size());
    std::vector<char> visiting(nodes.size(), 0);
    return BuildNode(nodes, built, visiting, 0);
}

// ---------------- decoding ----------------

namespace {

// Crop or zero-pad `wave` (starting at t_env) to [start, end); returns the effective start.
template <class T>
double TailorEnvWave(double t_env, std::vector<T>& wave, long long& size, double start, double end,
                     const WaveCache& wc) {
    double res = wc.resolution();
    double roa = static_cast<double>(wc.resolution_over_atol());
    double t_last = t_env + res * static_cast<double>(wave.size() - 1);
    double s = res * (std::round(start / kTimeAtol) / roa);
    if (s < t_env) {
        long long pad = std::llround((t_env - s) / res);
        wave.insert(wave.begin(), static_cast<size_t>(pad), T(0));
    } else if (s > t_last) {
        throw std::runtime_error("Input start > env.end()!");
    } else {
        long long drop = std::llround((s - t_env) / res);
        wave.erase(wave.begin(), wave.begin() + drop);
    }
    double e = res * (std::round(end / kTimeAtol) / roa + -1.0);
    size = static_cast<long long>(wave.size());
    if (e > t_last) {
        long long pad = std::llround((e - t_last) / res);
        wave.insert(wave.end(), static_cast<size_t>(pad), T(0));
        size = static_cast<long long>(wave.size());
    } else if (e < t_env) {
        throw std::runtime_error("Input end < env.start()!");
    } else {
        long long drop = std::llround((t_last - e) / res);
        size -= drop;
        if (size >= 0)
            wave.resize(static_cast<size_t>(size));
    }
    return s;
}

template <class T>
WavePtr<T> FinalWave(const EnvPtr& env, const TimeFunc<T>& tf) {
    // An EnvSum already returns a freshly summed buffer with unit amplitude.
    if (env->Type() == EnvType::kEnvSum)
        return tf.lazy->wave;
    return tf.lazy->Eval();
}

}  // namespace

Decoded DecodeEnvelope(const EnvPtr& env, WaveCache& wc) {
    Decoded d;
    if (env->IsComplex()) {
        auto tf = env->ComplexTimeFunc(wc, 0.0);
        d.t = tf.t;
        d.complex = FinalWave(env, tf);
        d.size = static_cast<long long>(d.complex->size());
    } else {
        auto tf = env->RealTimeFunc(wc, 0.0);
        d.t = tf.t;
        d.real = FinalWave(env, tf);
        d.size = static_cast<long long>(d.real->size());
    }
    return d;
}

Decoded DecodeEnvelope(const EnvPtr& env, WaveCache& wc, double start, double end) {
    Decoded d;
    if (env->IsComplex()) {
        auto tf = env->ComplexTimeFunc(wc, 0.0);
        d.complex = FinalWave(env, tf);
        d.t = TailorEnvWave(tf.t, *d.complex, d.size, start, end, wc);
    } else {
        auto tf = env->RealTimeFunc(wc, 0.0);
        d.real = FinalWave(env, tf);
        d.t = TailorEnvWave(tf.t, *d.real, d.size, start, end, wc);
    }
    return d;
}

}  // namespace quspect::native::pulse

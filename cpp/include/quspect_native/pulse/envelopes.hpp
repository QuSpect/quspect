// Control envelopes in the time domain for superconducting-qubit pulse sequences.
//
// Clean-room reimplementation of the C++ extension of envelopes-qc 0.1.5 (MIT). The numerical
// operation order follows the reference build so that decoded waveforms are bit-identical; the
// sources must be compiled without floating-point contraction (-ffp-contract=off).

#pragma once

#include <complex>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace quspect::native::pulse {

using CFloat = std::complex<float>;

constexpr double kTimeAtol = 1e-5;
constexpr double kFreqAtol = 1e-5;
constexpr double kFloatAtol = 1e-8;
constexpr double kMixDurationAtol = 1e-8;
constexpr long long kEndPadding = 2;

/// Serialization tags, also used as runtime type ids.
enum class EnvType : std::uint8_t {
    kGaussian = 0,
    kGaussianDrag = 1,
    kCosineDrag = 2,
    kTriangle = 3,
    kRect = 4,
    kFlattop = 5,
    kRippleRect = 6,
    kPowerSineCosine = 7,
    kGaussianXPower = 8,
    kCustom = 9,
    kMixExp = 10,
    kEnvMix = 11,
    kEnvShift = 12,
    kEnvProd = 13,
    kEnvSum = 14,
};

template <class T>
using WavePtr = std::shared_ptr<std::vector<T>>;

/// Cache key: integer-quantized envelope parameters.
struct EnvFeature {
    std::vector<long long> values;

    bool operator==(const EnvFeature& other) const {
        return values == other.values;
    }
    std::string ToString() const;

    struct Hash {
        std::size_t operator()(const EnvFeature& feature) const;
    };
};

template <class T>
using FeatureMap = std::unordered_map<EnvFeature, WavePtr<T>, EnvFeature::Hash>;
template <class T>
using BaseMap = std::unordered_map<long long, std::shared_ptr<FeatureMap<T>>>;
template <class T>
using NameMap = std::unordered_map<std::string, BaseMap<T>>;

/// Per-resolution cache of base waveforms shared by envelopes with equal parameters.
class WaveCache {
 public:
    explicit WaveCache(double resolution);

    double resolution() const {
        return resolution_;
    }
    long long resolution_over_atol() const {
        return resolution_over_atol_;
    }

    /// Returns (base_start, shift_t, feature map) for an envelope starting at `start`.
    template <class T>
    std::tuple<double, double, FeatureMap<T>*> GetCache(const std::string& name, double start);

    std::string ToString() const;

 private:
    template <class T>
    NameMap<T>& Maps();

    double resolution_;
    long long resolution_over_atol_;
    NameMap<float> real_cache_;
    NameMap<CFloat> complex_cache_;
};

/// wave * amp * extras[0] * extras[1] * ..., evaluated element-wise on demand.
template <class T>
struct LazyArray {
    T amp;
    WavePtr<T> wave;
    std::vector<WavePtr<T>> extras;

    LazyArray(T amp_value, WavePtr<T> wave_value) : amp(amp_value), wave(std::move(wave_value)) {
    }
    LazyArray& operator*=(const LazyArray& other);
    WavePtr<T> Eval() const;
};

template <class T>
struct TimeFunc {
    double t;
    std::shared_ptr<LazyArray<T>> lazy;
};

class AbstractEnvelope;
using EnvPtr = std::shared_ptr<AbstractEnvelope>;

class AbstractEnvelope : public std::enable_shared_from_this<AbstractEnvelope> {
 public:
    virtual ~AbstractEnvelope() = default;

    virtual EnvType Type() const = 0;
    /// Distinguishes COMPLEXEnvProd from RealEnvProd (both tagged EnvType::kEnvProd).
    virtual bool IsComplexProd() const {
        return false;
    }
    virtual double Start() const = 0;
    virtual double End() const = 0;
    double Duration() const {
        return End() - Start();
    }
    virtual bool IsComplex() const = 0;

    virtual TimeFunc<float> RealTimeFunc(WaveCache& wc, double dt) const = 0;
    virtual TimeFunc<CFloat> ComplexTimeFunc(WaveCache& wc, double dt) const = 0;

    virtual std::string ToString() const = 0;
    /// Type-specific payload written after the tag byte.
    virtual void SerializeState(std::string& out) const = 0;
    virtual std::vector<EnvPtr> Children() const {
        return {};
    }
    /// Shallow clone that takes (possibly copied) children in Children() order.
    virtual EnvPtr Rebuild(const std::vector<EnvPtr>& children) const = 0;
};

// ---------------- leaf envelopes ----------------

class Envelope : public AbstractEnvelope {
 public:
    double Start() const override {
        return start_;
    }
    double End() const override {
        return end_;
    }
    bool IsComplex() const override {
        return complex_;
    }
    float amp() const {
        return amp_;
    }

    TimeFunc<float> RealTimeFunc(WaveCache& wc, double dt) const override;
    TimeFunc<CFloat> ComplexTimeFunc(WaveCache& wc, double dt) const override;
    std::string ToString() const override {
        return ClassName();
    }

    virtual std::string ClassName() const = 0;
    virtual std::string CacheName() const {
        return ClassName();
    }

 protected:
    Envelope(double start, double end, float amp, bool is_complex, std::vector<long long> feature);

    /// Base waveform with the envelope starting at `base_start` within [0, res).
    virtual WavePtr<float> RealModel(double base_start, std::size_t n, double res) const;
    virtual WavePtr<CFloat> ComplexModel(double base_start, std::size_t n, double res) const;

    double start_;
    double end_;
    float amp_;
    bool complex_;
    EnvFeature feature_;
};

class Gaussian : public Envelope {
 public:
    Gaussian(double t0, double w, float amp);
    EnvType Type() const override {
        return EnvType::kGaussian;
    }
    std::string ClassName() const override {
        return "Gaussian";
    }
    void SerializeState(std::string& out) const override;
    EnvPtr Rebuild(const std::vector<EnvPtr>& children) const override;

 protected:
    WavePtr<float> RealModel(double base_start, std::size_t n, double res) const override;

 private:
    double t0_, w_, sigma_;
};

class GaussianDrag : public Envelope {
 public:
    GaussianDrag(double t0, double w, float amp, double coef, double df, double phase);
    EnvType Type() const override {
        return EnvType::kGaussianDrag;
    }
    std::string ClassName() const override {
        return "GaussianDRAG";
    }
    TimeFunc<CFloat> ComplexTimeFunc(WaveCache& wc, double dt) const override;
    void SerializeState(std::string& out) const override;
    EnvPtr Rebuild(const std::vector<EnvPtr>& children) const override;

 protected:
    WavePtr<CFloat> ComplexModel(double base_start, std::size_t n, double res) const override;

 private:
    double t0_, w_, sigma_, coef_, df_, phase_;
};

class CosineDrag : public Envelope {
 public:
    CosineDrag(double t0, double w, float amp, double coef, double df, double phase);
    EnvType Type() const override {
        return EnvType::kCosineDrag;
    }
    std::string ClassName() const override {
        return "CosineDRAG";
    }
    TimeFunc<CFloat> ComplexTimeFunc(WaveCache& wc, double dt) const override;
    void SerializeState(std::string& out) const override;
    EnvPtr Rebuild(const std::vector<EnvPtr>& children) const override;

 protected:
    WavePtr<CFloat> ComplexModel(double base_start, std::size_t n, double res) const override;

 private:
    double t0_, w_, coef_, df_, phase_;
};

class Triangle : public Envelope {
 public:
    Triangle(double t0, double tlen, float amp, bool fall);
    EnvType Type() const override {
        return EnvType::kTriangle;
    }
    std::string ClassName() const override {
        return "Triangle";
    }
    void SerializeState(std::string& out) const override;
    EnvPtr Rebuild(const std::vector<EnvPtr>& children) const override;

 protected:
    WavePtr<float> RealModel(double base_start, std::size_t n, double res) const override;

 private:
    double t0_, tlen_;
    bool fall_;
};

class Rect : public Envelope {
 public:
    Rect(double t0, double tlen, float amp);
    EnvType Type() const override {
        return EnvType::kRect;
    }
    std::string ClassName() const override {
        return "Rect";
    }
    void SerializeState(std::string& out) const override;
    EnvPtr Rebuild(const std::vector<EnvPtr>& children) const override;

 protected:
    WavePtr<float> RealModel(double base_start, std::size_t n, double res) const override;

 private:
    double t0_, tlen_;
};

class Flattop : public Envelope {
 public:
    Flattop(double t0, double tlen, double w_left, double w_right, float amp);
    EnvType Type() const override {
        return EnvType::kFlattop;
    }
    std::string ClassName() const override {
        return "Flattop";
    }
    void SerializeState(std::string& out) const override;
    EnvPtr Rebuild(const std::vector<EnvPtr>& children) const override;

 protected:
    WavePtr<float> RealModel(double base_start, std::size_t n, double res) const override;

 private:
    double t0_, tlen_, w_left_, w_right_, a_left_, a_right_;
};

class RippleRect : public Envelope {
 public:
    RippleRect(double t0, double tlen, float amp, double w, double ripple0, double ripple1, double ripple2,
               double ripple3);
    EnvType Type() const override {
        return EnvType::kRippleRect;
    }
    std::string ClassName() const override {
        return "RippleRect";
    }
    void SerializeState(std::string& out) const override;
    EnvPtr Rebuild(const std::vector<EnvPtr>& children) const override;

 protected:
    WavePtr<float> RealModel(double base_start, std::size_t n, double res) const override;

 private:
    double t0_, tlen_, w_;
    double ripples_[4];
};

class PowerSineCosine : public Envelope {
 public:
    PowerSineCosine(double t0, double tlen, float amp, int m, int n, double k, double phase);
    EnvType Type() const override {
        return EnvType::kPowerSineCosine;
    }
    std::string ClassName() const override {
        return "PowerSineCosine";
    }
    void SerializeState(std::string& out) const override;
    EnvPtr Rebuild(const std::vector<EnvPtr>& children) const override;

 protected:
    WavePtr<float> RealModel(double base_start, std::size_t n, double res) const override;

 private:
    double t0_, tlen_;
    int m_, n_;
    double k_, phase_;
};

class GaussianXPower : public Envelope {
 public:
    GaussianXPower(double t0, double w, float amp, int m);
    EnvType Type() const override {
        return EnvType::kGaussianXPower;
    }
    std::string ClassName() const override {
        return "GaussianXPower";
    }
    void SerializeState(std::string& out) const override;
    EnvPtr Rebuild(const std::vector<EnvPtr>& children) const override;

 protected:
    WavePtr<float> RealModel(double base_start, std::size_t n, double res) const override;

 private:
    double t0_, w_, sigma_;
    int m_;
};

/// Arbitrary sampled waveform, linearly interpolated onto the cache grid.
class Custom : public Envelope {
 public:
    Custom(double start, double dt, WavePtr<float> data);
    Custom(double start, double dt, WavePtr<CFloat> data);
    EnvType Type() const override {
        return EnvType::kCustom;
    }
    std::string ClassName() const override {
        return "Custom";
    }
    std::string CacheName() const override {
        return ToString();
    }
    std::string ToString() const override;
    void SerializeState(std::string& out) const override;
    EnvPtr Rebuild(const std::vector<EnvPtr>& children) const override;

 protected:
    WavePtr<float> RealModel(double base_start, std::size_t n, double res) const override;
    WavePtr<CFloat> ComplexModel(double base_start, std::size_t n, double res) const override;

 private:
    void InitId();

    double dt_;
    WavePtr<float> real_data_;
    WavePtr<CFloat> complex_data_;
    unsigned char id_[32];
};

// ---------------- composite envelopes ----------------

/// Carrier exp(-2*pi*i*df*t + i*phase) used by EnvMix.
class MixExp : public AbstractEnvelope {
 public:
    MixExp(double start, double end, double df, double phase, bool dynamical);
    EnvType Type() const override {
        return EnvType::kMixExp;
    }
    double Start() const override {
        return start_;
    }
    double End() const override {
        return end_;
    }
    bool IsComplex() const override {
        return true;
    }
    TimeFunc<float> RealTimeFunc(WaveCache& wc, double dt) const override;
    TimeFunc<CFloat> ComplexTimeFunc(WaveCache& wc, double dt) const override;
    std::string ToString() const override {
        return "MixExp";
    }
    void SerializeState(std::string& out) const override;
    EnvPtr Rebuild(const std::vector<EnvPtr>& children) const override;

 private:
    double start_, end_, df_, phase_;
    bool dynamical_;
    EnvFeature feature_;
};

class EnvMix : public AbstractEnvelope {
 public:
    EnvMix(EnvPtr env, double df, double phase, bool dynamical);
    EnvMix(EnvPtr env, std::shared_ptr<MixExp> carrier) : env_(std::move(env)), carrier_(std::move(carrier)) {
    }
    EnvType Type() const override {
        return EnvType::kEnvMix;
    }
    double Start() const override {
        return env_->Start();
    }
    double End() const override {
        return env_->End();
    }
    bool IsComplex() const override {
        return true;
    }
    TimeFunc<float> RealTimeFunc(WaveCache& wc, double dt) const override;
    TimeFunc<CFloat> ComplexTimeFunc(WaveCache& wc, double dt) const override;
    std::string ToString() const override {
        return "EnvMix(" + env_->ToString() + ")";
    }
    void SerializeState(std::string& /*out*/) const override {
    }
    std::vector<EnvPtr> Children() const override {
        return {env_, carrier_};
    }
    EnvPtr Rebuild(const std::vector<EnvPtr>& children) const override;

 private:
    EnvPtr env_;
    std::shared_ptr<MixExp> carrier_;
};

class EnvShift : public AbstractEnvelope {
 public:
    EnvShift(EnvPtr env, double dt) : env_(std::move(env)), dt_(dt) {
    }
    EnvType Type() const override {
        return EnvType::kEnvShift;
    }
    double Start() const override {
        return env_->Start() + dt_;
    }
    double End() const override {
        return env_->End() + dt_;
    }
    bool IsComplex() const override {
        return env_->IsComplex();
    }
    TimeFunc<float> RealTimeFunc(WaveCache& wc, double dt) const override {
        return env_->RealTimeFunc(wc, dt_ + dt);
    }
    TimeFunc<CFloat> ComplexTimeFunc(WaveCache& wc, double dt) const override {
        return env_->ComplexTimeFunc(wc, dt_ + dt);
    }
    std::string ToString() const override {
        return "EnvShift(" + env_->ToString() + ")";
    }
    void SerializeState(std::string& out) const override;
    std::vector<EnvPtr> Children() const override {
        return {env_};
    }
    EnvPtr Rebuild(const std::vector<EnvPtr>& children) const override;

    const EnvPtr& env() const {
        return env_;
    }
    double dt() const {
        return dt_;
    }

 private:
    EnvPtr env_;
    double dt_;
};

/// Envelope multiplied by a constant; T is float (RealEnvProd) or CFloat (COMPLEXEnvProd).
template <class T>
class EnvProd : public AbstractEnvelope {
 public:
    EnvProd(EnvPtr env, T c) : env_(std::move(env)), c_(c) {
    }
    EnvType Type() const override {
        return EnvType::kEnvProd;
    }
    bool IsComplexProd() const override {
        return std::is_same_v<T, CFloat>;
    }
    double Start() const override {
        return env_->Start();
    }
    double End() const override {
        return env_->End();
    }
    bool IsComplex() const override;
    TimeFunc<float> RealTimeFunc(WaveCache& wc, double dt) const override;
    TimeFunc<CFloat> ComplexTimeFunc(WaveCache& wc, double dt) const override;
    std::string ToString() const override {
        return "EnvProd(" + env_->ToString() + ")";
    }
    void SerializeState(std::string& out) const override;
    std::vector<EnvPtr> Children() const override {
        return {env_};
    }
    EnvPtr Rebuild(const std::vector<EnvPtr>& children) const override;

    const EnvPtr& env() const {
        return env_;
    }
    T constant() const {
        return c_;
    }

 private:
    EnvPtr env_;
    T c_;
};

class EnvSum : public AbstractEnvelope {
 public:
    EnvSum() = default;
    explicit EnvSum(std::vector<EnvPtr> items) : items_(std::move(items)) {
    }
    EnvType Type() const override {
        return EnvType::kEnvSum;
    }
    double Start() const override;
    double End() const override;
    bool IsComplex() const override;
    TimeFunc<float> RealTimeFunc(WaveCache& wc, double dt) const override;
    TimeFunc<CFloat> ComplexTimeFunc(WaveCache& wc, double dt) const override;
    std::string ToString() const override;
    void SerializeState(std::string& out) const override;
    std::vector<EnvPtr> Children() const override {
        return items_;
    }
    EnvPtr Rebuild(const std::vector<EnvPtr>& children) const override;

    const std::vector<EnvPtr>& items() const {
        return items_;
    }
    /// Appends `env`, flattening EnvSum operands (safe when `env` is this sum).
    void Append(const EnvPtr& env);

 private:
    template <class T>
    TimeFunc<T> SumTimeFunc(WaveCache& wc, double dt) const;

    std::vector<EnvPtr> items_;
};

// ---------------- operations ----------------

EnvPtr Add(const EnvPtr& a, const EnvPtr& b);
EnvPtr Add(const EnvPtr& a, double c);
EnvPtr InplaceAdd(const EnvPtr& a, const EnvPtr& b);
EnvPtr Multiply(const EnvPtr& a, float c);
EnvPtr Multiply(const EnvPtr& a, CFloat c);
EnvPtr Shift(const EnvPtr& a, double dt);
EnvPtr Mix(const EnvPtr& env, double df, double phase, bool dynamical);
EnvPtr Align(const EnvPtr& env, const std::vector<double>& dt);
EnvPtr Align(const EnvPtr& env, const std::vector<double>& dt, const std::vector<float>& amp);
EnvPtr Align(const EnvPtr& env, const std::vector<double>& dt, const std::vector<CFloat>& amp);
std::vector<EnvPtr> Split(const EnvPtr& env, const std::vector<double>& starts, const std::vector<double>& ends);
EnvPtr Copy(const EnvPtr& env);

std::string Serialize(const EnvPtr& env);
EnvPtr Deserialize(const std::string& data);

/// Decoded waveform: (start time, samples). Exactly one of real/complex is set.
struct Decoded {
    double t = 0.0;
    WavePtr<float> real;
    WavePtr<CFloat> complex;
    /// Number of samples; negative when the requested window is inverted.
    long long size = 0;
};

Decoded DecodeEnvelope(const EnvPtr& env, WaveCache& wc);
Decoded DecodeEnvelope(const EnvPtr& env, WaveCache& wc, double start, double end);

}  // namespace quspect::native::pulse

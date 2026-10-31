// Docstrings of the envelopes-qc 0.1.5 extension (MIT), kept verbatim for API compatibility.
#pragma once

namespace quspect::native::pulse::docs {

inline constexpr const char* kModule = R"__doc__(Envelops impelemented in C++.)__doc__";
inline constexpr const char* kCosineDRAGClass = R"__doc__(Cosine envelope shaped with DRAG.)__doc__";
inline constexpr const char* kCosineDRAGInit = R"__doc__(Parameters:
t0 is the center of Cosine, w is the width of Cosine, and amp is the amplitude.
Mathmatical formula: [Cosine(t0, w, amp) + i * coef * CosineDeriv(t0, w, amp)] * exp(i * (-2* pi * df * t + phase)))__doc__";
inline constexpr const char* kCustomClass = R"__doc__(Custom envelope from arbitrary waveform data.)__doc__";
inline constexpr const char* kCustomInit = R"__doc__(Custom envelope from arbitrary waveform.
Parameters:
start is the start time, dt is the sample interval of the input data, and data is a 1-D numeric array (float32/float64/complex64/complex128 and other integer or boolean dtypes are accepted via promotion).)__doc__";
inline constexpr const char* kFlattopClass = R"__doc__(Rectangular envelope convolved with Gaussian kernel.)__doc__";
inline constexpr const char* kFlattopInit = R"__doc__(Parameters:
t0 is the start of Rect, tlen is the duriation, and amp is the amplitude.
w_left(w_right) is the width of Gaussian kernel to convolve the left(right) edge of Rect.)__doc__";
inline constexpr const char* kGaussianClass = R"__doc__(Gaussian envelope.)__doc__";
inline constexpr const char* kGaussianInit = R"__doc__(Parameters:
t0 is the center of Gaussian, w is the width at half maximum, and amp is the amplitude.)__doc__";
inline constexpr const char* kGaussianDRAGClass = R"__doc__(Gaussian envelope shaped with DRAG.)__doc__";
inline constexpr const char* kGaussianDRAGInit = R"__doc__(Parameters:
t0 is the center of Gaussian, w is the width at half maximum, and amp is the amplitude.
Mathmatical formula: [Gaussin(t0, w, amp) + i * coef * GaussinDeriv(t0, w, amp)] * exp(i * (-2* pi * df * t + phase)))__doc__";
inline constexpr const char* kGaussianXPowerClass = R"__doc__(Gaussian envelope multiplied by x^m.)__doc__";
inline constexpr const char* kGaussianXPowerInit = R"__doc__(Parameters:
t0 is the center, w is the width at half maximum, amp is the amplitude.
Mathmatical formula: Gaussian(t0, w) * x^m)__doc__";
inline constexpr const char* kPowerSineCosineClass = R"__doc__(Envelope with sin and cos power functions.)__doc__";
inline constexpr const char* kPowerSineCosineInit = R"__doc__(Parameters:
t0 is the start, tlen is the duration, amp is the amplitude.
Mathmatical formula: (sin(k*pi*t/T+phase)^m)*(cos(k*pi*t/T+phase)^n))__doc__";
inline constexpr const char* kRectClass = R"__doc__(Rectangle envelope.)__doc__";
inline constexpr const char* kRectInit = R"__doc__(Parameters:
t0 is the start of Rectangle, tlen is the duriation, and amp is the amplitude.)__doc__";
inline constexpr const char* kRippleRectClass
    = R"__doc__(Rectangular envelope with ripples convolved with Gaussian kernel.)__doc__";
inline constexpr const char* kRippleRectInit = R"__doc__(Parameters:
t0 is the start of Rect, tlen is the duriation, and amp is the amplitude.
ripple0, ripple1, ripple2, ripple3 describe the ripples of the Rect.
w is the width of Gaussian kernel to convolve the Rect.)__doc__";
inline constexpr const char* kTriangleClass = R"__doc__(Triangle envelope.)__doc__";
inline constexpr const char* kTriangleInit = R"__doc__(Parameters:
t0 is the start of Triangle, tlen is the duriation, amp is the amplitude and fall indicates whether the Triangle is falling or rising.)__doc__";
inline constexpr const char* kWaveCacheClass
    = R"__doc__(Cache for waveforms. The same envelope might yield both real and complex wave data which will be cached separately.)__doc__";
inline constexpr const char* kWaveCacheResolution = R"__doc__(Get the resolution of the cache.)__doc__";
inline constexpr const char* kAlign0 = R"__doc__(FAST aligning the envelope to the given time shifts.)__doc__";
inline constexpr const char* kAlign1
    = R"__doc__(FAST aligning the envelope to the given time shifts and real amplitudes.)__doc__";
inline constexpr const char* kAlign2
    = R"__doc__(FAST aligning the envelope to the given time shifts and complex amplitudes.)__doc__";
inline constexpr const char* kDecodeEnvelope0
    = R"__doc__(Decodes the envelope and returns (start_time, wave) where wave is a numpy array (zero-copy view of the internal buffer).
Start and end are extracted from env.start() and env.end().)__doc__";
inline constexpr const char* kDecodeEnvelope1
    = R"__doc__(Decodes the envelope and returns (start_time, wave) where wave is a numpy array (zero-copy view of the internal buffer).)__doc__";
inline constexpr const char* kDeserialization0 = R"__doc__(Serializes the envelope to support pickle.loads.)__doc__";
inline constexpr const char* kMix0 = R"__doc__(FAST mixing the envelope by the given df and phase.
dynamical=True means phase keeps 0 at t=0 (i.e. env * exp(i * (-2* pi * df * t + phase)).
dynamical=False means phase keeps 0 at t=dt(env>>dt) (i.e. env * exp(i * (-2* pi * df * (t - dt) + phase)).)__doc__";
inline constexpr const char* kSerialization0 = R"__doc__(Serializes the envelope to support pickle.dumps.)__doc__";
inline constexpr const char* kSplit0 = R"__doc__(Split env according to starts and ends.
The provided starts and ends must be pre-sorted and have the same length.)__doc__";

}  // namespace quspect::native::pulse::docs

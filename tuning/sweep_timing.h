// Timing every backend at one point, shared by sweep_eigh.cpp and
// sweep_svd.cpp (sweep_qr.cpp times one backend a process).
//
// Two rounds. First each backend in turn: its correctness run (which compiles
// the pipelines and makes the workspaces), a warm-up unless that run's call
// took 100 ms or more (whose own length settles the clocks: two calls, or one
// taking 20 ms or more), and one timed call. Then each again: at least five
// samples (three for calls over 300 ms), then until 150 ms is spent, at most
// 25; except that a backend whose first sample took 20 ms or more and more
// than kLoserRatio times the fastest first sample of its mode (with vectors,
// or values alone) stops at two. Only a point's winner and near-winners need
// tight times: the analysis reads medians, and its tests for a clear winner
// compare the fastest with the next within a mode. Shorter calls keep every
// sample: two of a call under a millisecond, taken after other backends ran,
// read up to 2x slow, and the whole budget costs little.
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <map>
#include <vector>

namespace sweep {

struct Timing {
    double median = 0, p25 = 0, p75 = 0;
    int reps = 0;
};

struct Backend {
    // The correctness run: the error (infinity where the backend cannot run
    // here), and the call's own time in call_ms.
    std::function<float(double& call_ms)> check;
    std::function<void()> run;   // one call, waited for
    int mode = 0;                // compared only with backends of its mode
};

struct Result {
    bool ok = false;
    Timing t;
};

constexpr double kLoserRatio = 1.3;
constexpr double kLoserMinMs = 20.0;
constexpr float kTolerance = 1e-3f;

inline double ms_since(std::chrono::high_resolution_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
}

inline double quantile(const std::vector<double>& sorted, double q) {
    if (sorted.empty()) return 0.0;
    const double pos = q * (sorted.size() - 1);
    const size_t lo = (size_t)std::floor(pos), hi = (size_t)std::ceil(pos);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

inline double timed(const Backend& b) {
    const auto t0 = std::chrono::high_resolution_clock::now();
    b.run();
    return ms_since(t0);
}

inline std::vector<Result> measure_all(const std::vector<Backend>& bs,
                                       double budget_ms = 150.0, int max_reps = 25) {
    std::vector<Result> out(bs.size());
    std::vector<std::vector<double>> samples(bs.size());
    std::map<int, double> fastest;   // a mode's fastest first sample
    for (size_t i = 0; i < bs.size(); ++i) {
        double call_ms = 0.0;
        const float err = bs[i].check(call_ms);
        out[i].ok = std::isfinite(err) && err <= kTolerance;
        if (!out[i].ok) continue;
        if (call_ms < 100.0)
            for (int w = 0; w < 2; ++w)
                if (timed(bs[i]) >= 20.0) break;
        samples[i].push_back(timed(bs[i]));
        auto f = fastest.find(bs[i].mode);
        if (f == fastest.end() || samples[i][0] < f->second) fastest[bs[i].mode] = samples[i][0];
    }
    for (size_t i = 0; i < bs.size(); ++i) {
        if (!out[i].ok) continue;
        std::vector<double>& s = samples[i];
        const bool loser = s[0] >= kLoserMinMs && s[0] > kLoserRatio * fastest[bs[i].mode];
        if (s[0] < 100.0)   // warm again, as in the first round: other backends ran since
            for (int w = 0; w < 2; ++w)
                if (timed(bs[i]) >= 20.0) break;
        double total = s[0];
        int min_reps = loser ? 2 : s[0] > 300.0 ? 3 : 5;
        const int cap = loser ? 2 : max_reps;
        while ((int)s.size() < cap && (total < budget_ms || (int)s.size() < min_reps)) {
            const double dt = timed(bs[i]);
            s.push_back(dt);
            total += dt;
            if (dt > 300.0) min_reps = std::min(min_reps, 3);   // a slow call is its own evidence
        }
        std::sort(s.begin(), s.end());
        out[i].t = Timing{quantile(s, 0.50), quantile(s, 0.25), quantile(s, 0.75), (int)s.size()};
    }
    return out;
}

} // namespace sweep

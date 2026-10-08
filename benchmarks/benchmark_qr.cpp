#include <algorithm>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <vector>
#include <cmath>
#include <random>
#include <cstdlib>
#include <string>

#include <mlx/mlx.h>
#include <mlx/linalg.h>

#include <metal_linalg/c_api.h>   // metal_linalg_qr_backend, for the backend's name
#include <metal_linalg/qr.h>

using namespace mlx::core;

// =============================================================================
// Config
// =============================================================================

struct BenchConfig {
    int M, N;
    int max_batch = 0;  // 0 = no limit; set to skip batches above this value
};

static const std::vector<BenchConfig> SMALL_CONFIGS = {
    {8, 8},
    {16, 16},
    {32, 32},
    { 64,  64},
    {128,  64},
    {256, 128, 1000},   // skipped for batch > 1000
    {512, 256, 500},   // skipped for batch > 500
};

static const std::vector<BenchConfig> LARGE_CONFIGS = {
    { 512,  512},
    {1024,  512, 17},
    {5000, 5000, 9},
};

static const std::vector<int> SMALL_BATCHES = {10, 50, 100, 500, 1000, 5000, 10000, 15000};
static const std::vector<int> LARGE_BATCHES = {1, 8, 16, 32};

// =============================================================================
// Helpers
// =============================================================================

static array random_matrix(int batch, int M, int N, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> data(batch * M * N);
    for (auto& v : data) v = dist(rng);
    if (batch == 1)
        return array(data.begin(), {M, N}, float32);
    return array(data.begin(), {batch, M, N}, float32);
}

static array slice_batch(const array& A, int b, int M, int N) {
    return reshape(slice(A, {b, 0, 0}, {b + 1, M, N}), {M, N});
}

static float reconstruction_error(const array& Q, const array& R, const array& A) {
    array QR       = matmul(Q, R);
    array diff     = subtract(QR, A);
    array sq       = multiply(diff, diff);
    array mean_err = mean(sqrt(sum(sq, {-1, -2})));
    eval({mean_err});
    return mean_err.item<float>();
}

template<typename Fn>
static double time_ms(Fn fn, int warmup = 2, int reps = 5) {
    for (int i = 0; i < warmup; ++i) fn();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < reps; ++i) fn();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
}

static bool exceeds_batch_limit(const BenchConfig& cfg, int batch) {
    return cfg.max_batch > 0 && batch > cfg.max_batch;
}

// =============================================================================
// Result collection
// =============================================================================

struct BenchResult {
    double gpu_ms = 0;
    double cpu_ms = 0;
    float  err_gpu = 0;
    bool   skipped = false;
};

static BenchResult run_benchmark(int batch, int M, int N) {
    BenchResult r;

    array A = random_matrix(batch, M, N);
    eval({A});

    // --- GPU ---
    set_default_device(Device::gpu);
    r.gpu_ms = time_ms([&] {
        auto [Q, R] = metal_linalg::qr_accelerated(A);
        eval({Q, R});
    });
    auto [Q_gpu, R_gpu] = metal_linalg::qr_accelerated(A);
    eval({Q_gpu, R_gpu});

    // --- CPU (MLX / LAPACK) ---
    set_default_device(Device::cpu);
    r.cpu_ms = time_ms([&] {
        if (batch == 1) {
            auto [Q, R] = linalg::qr(A, Device::cpu);
            eval({Q, R});
        } else {
            for (int b = 0; b < batch; ++b) {
                auto [Q, R] = linalg::qr(slice_batch(A, b, M, N), Device::cpu);
                eval({Q, R});
            }
        }
    });

    // --- Error ---
    set_default_device(Device::gpu);
    r.err_gpu = reconstruction_error(Q_gpu, R_gpu, A);

    return r;
}

// =============================================================================
// Table rendering
// =============================================================================

static const int SHAPE_W = 14;   // width of the "Shape" column
static const int CELL_W  = 17;   // width of each batch-size column

static std::string pad(const std::string& s, int width, bool left = false) {
    if ((int)s.size() >= width) return s.substr(0, width);
    std::string out = left ? s : std::string(width - s.size(), ' ') + s;
    if (left) out += std::string(width - s.size(), ' ');
    return out;
}

static std::string format_shape(int M, int N) {
    std::ostringstream ss;
    ss << std::setw(4) << M << " x " << std::setw(4) << N;
    return ss.str();
}

static std::string format_ms(double ms) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2) << ms << " ms";
    return ss.str();
}

static std::string format_error(float err) {
    std::ostringstream ss;
    ss << std::scientific << std::setprecision(2) << err;
    return ss.str();
}

static std::string format_speedup(double speedup) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2) << speedup << "x";
    return ss.str();
}

static std::string make_divider(int n_batches) {
    std::string s = std::string(SHAPE_W, '-') + "+";
    for (int i = 0; i < n_batches; ++i)
        s += std::string(CELL_W, '-') + "+";
    return s;
}

static void print_table(
    const std::string& title,
    const std::vector<BenchConfig>& configs,
    const std::vector<int>& batches,
    const std::vector<std::vector<BenchResult>>& results  // [config_idx][batch_idx]
) {
    const int nb = (int)batches.size();

    std::cout << "\n[ " << title << " ]\n\n";

    // Header
    std::cout << pad("Shape", SHAPE_W, true) << "|";
    for (int b : batches)
        std::cout << pad("batch=" + std::to_string(b), CELL_W) << "|";
    std::cout << "\n";

    std::cout << make_divider(nb) << "\n";

    // Five sub-rows per config: shape header, GPU time, CPU time, speedup, error
    for (int ci = 0; ci < (int)configs.size(); ++ci) {
        const auto& cfg = configs[ci];

        // Sub-row 1: matrix shape (no data — acts as group header)
        std::cout << pad(format_shape(cfg.M, cfg.N), SHAPE_W, true) << "|";
        for (int bi = 0; bi < nb; ++bi)
            std::cout << std::string(CELL_W, ' ') << "|";
        std::cout << "\n";

        // Sub-row 2: GPU time
        std::cout << pad("  GPU", SHAPE_W, true) << "|";
        for (int bi = 0; bi < nb; ++bi) {
            const auto& r = results[ci][bi];
            std::cout << pad(r.skipped ? "--" : format_ms(r.gpu_ms), CELL_W) << "|";
        }
        std::cout << "\n";

        // Sub-row 3: CPU time
        std::cout << pad("  CPU", SHAPE_W, true) << "|";
        for (int bi = 0; bi < nb; ++bi) {
            const auto& r = results[ci][bi];
            std::cout << pad(r.skipped ? "--" : format_ms(r.cpu_ms), CELL_W) << "|";
        }
        std::cout << "\n";

        // Sub-row 4: speedup
        std::cout << pad("  Speedup", SHAPE_W, true) << "|";
        for (int bi = 0; bi < nb; ++bi) {
            const auto& r = results[ci][bi];
            std::cout << pad(r.skipped ? "--" : format_speedup(r.cpu_ms / r.gpu_ms), CELL_W) << "|";
        }
        std::cout << "\n";

        // Sub-row 5: reconstruction error
        std::cout << pad("  Error", SHAPE_W, true) << "|";
        for (int bi = 0; bi < nb; ++bi) {
            const auto& r = results[ci][bi];
            std::cout << pad(r.skipped ? "--" : format_error(r.err_gpu), CELL_W) << "|";
        }
        std::cout << "\n";

        std::cout << make_divider(nb) << "\n";
    }

    std::cout << "  Skipped cells (--) exceeded the per-config batch limit.\n";
}

// =============================================================================
// Modes (--modes): R alone and the complete Q against the reduced factors
// =============================================================================

template<typename Fn>
static double median_ms(Fn fn, int warmup = 3, int reps = 15) {
    for (int i = 0; i < warmup; ++i) fn();
    std::vector<double> t;
    for (int i = 0; i < reps; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        fn();
        t.push_back(std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count());
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

// The routed call in each mode, the shapes of docs/qr.md's table.
static void run_modes() {
    struct Shape { int batch, M, N; };
    const std::vector<Shape> shapes = {
        {4096, 32, 32}, {4096, 64, 64}, {1024, 128, 128}, {256, 256, 256}, {16, 1024, 1024},
        {1, 4096, 4096}, {1, 8192, 512}, {1, 512, 512}, {1, 128, 128}, {1, 256, 256}, {1, 64, 2048},
    };
    const char* modes[] = {"reduced", "r", "complete"};
    std::cout << "\n[ Modes: the routed call, median of 15 (3 warm-up discarded) ]\n\n"
              << pad("batch x M x N", 20, true) << pad("backend", 19, true)
              << pad("reduced", 12) << pad("r", 12) << pad("complete", 12) << pad("r faster by", 13) << "\n";
    set_default_device(Device::gpu);
    for (const Shape& s : shapes) {
        array A = random_matrix(s.batch, s.M, s.N);
        eval({A});
        double t[3];
        for (int i = 0; i < 3; ++i)
            t[i] = median_ms([&] {
                auto [Q, R] = metal_linalg::qr_accelerated(A, modes[i]);
                eval({Q, R});
            });
        std::ostringstream shape;
        shape << s.batch << " x " << s.M << " x " << s.N;
        std::cout << pad(shape.str(), 20, true) << pad(metal_linalg_qr_backend(s.M, s.N, s.batch), 19, true)
                  << pad(format_ms(t[0]), 12) << pad(format_ms(t[1]), 12) << pad(format_ms(t[2]), 12)
                  << pad(format_speedup(t[0] / t[1]), 13) << "\n";
        mlx::core::clear_cache();
    }
    std::cout << "\n";
}

// =============================================================================
// main
// =============================================================================

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [--modes]\n\n"
              << "  --modes  time R alone and the complete Q against the reduced factors\n"
              << "           (the routed call; docs/qr.md's table), instead of the tables below\n\n"
              << "  Per-config batch limits are set in SMALL_CONFIGS / LARGE_CONFIGS\n"
              << "  at the top of benchmark_qr.cpp via the max_batch field.\n"
              << "  Set max_batch = 0 on any config to run it at all batch sizes.\n";
}

int main(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (std::string(argv[i]) == "--modes") {
            run_modes();
            return 0;
        } else {
            std::cerr << "Unknown argument: " << argv[i] << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    std::cout << "\n";
    std::cout << "================================================================\n";
    std::cout << "          QR Decomposition: GPU (Metal) vs CPU (MLX)\n";
    std::cout << "================================================================\n\n";
    std::cout << "Timing     : average of 5 runs (2 warmup discarded)\n";
    std::cout << "Error      : mean ||Q*R - A||_F across batch\n";

    // -------------------------------------------------------------------------
    // Collect results
    // -------------------------------------------------------------------------
    auto collect = [&](const std::vector<BenchConfig>& configs,
                       const std::vector<int>& batches) {
        std::vector<std::vector<BenchResult>> results(
            configs.size(), std::vector<BenchResult>(batches.size()));
        // MLX's buffer cache left on, as an MLX program has it (see tuning/sweep_qr.cpp).

        for (int ci = 0; ci < (int)configs.size(); ++ci) {
            for (int bi = 0; bi < (int)batches.size(); ++bi) {
                int M = configs[ci].M, N = configs[ci].N, B = batches[bi];
                if (exceeds_batch_limit(configs[ci], B)) {
                    results[ci][bi].skipped = true;
                } else {
                    std::cout << "  running " << B << " x " << M << " x " << N << " ...\r" << std::flush;
                    results[ci][bi] = run_benchmark(B, M, N);
                    mlx::core::clear_cache();
                }
            }
        }
        std::cout << std::string(50, ' ') << "\r";  // clear progress line
        return results;
    };

    std::cout << "Running large benchmarks...\n";
    auto large_results = collect(LARGE_CONFIGS, LARGE_BATCHES);
    print_table("Large dimensions  —  M >= 512", LARGE_CONFIGS, LARGE_BATCHES, large_results);

    std::cout << "\nRunning small benchmarks...\n";
    auto small_results = collect(SMALL_CONFIGS, SMALL_BATCHES);
    print_table("Small dimensions  —  M < 512",  SMALL_CONFIGS, SMALL_BATCHES, small_results);

    std::cout << "\n";
    return 0;
}

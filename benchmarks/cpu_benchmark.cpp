// CPU baselines for the two things worth benchmarking against a future CUDA
// port: the full sequential batch pipeline (process_batch) and the
// embarrassingly-parallel piece of it (vec3_norm_batch), which is what
// benchmarks/gpu_benchmark.cu implements on the GPU for a direct comparison.
#include <benchmark/benchmark.h>

#include <cmath>
#include <vector>

#include "imu_math.hpp"
#include "imu_processor.hpp"

namespace {

imu::IMUSampleBatch make_synthetic_batch(std::size_t n) {
    imu::IMUSampleBatch batch(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double t = static_cast<double>(i) * 0.01;  // simulated 100 Hz stream
        auto& fr = batch[i];
        fr.t_ms = static_cast<std::int64_t>(t * 1000.0);
        fr.a1 = {0.1 * std::sin(t), 0.1 * std::cos(t), 1.0};
        fr.g1 = {std::sin(t) * 5.0, std::cos(t) * 5.0, 0.0};
        fr.a2 = {0.05 * std::sin(t * 0.5), 0.05 * std::cos(t * 0.5), 1.0};
        fr.g2 = {std::sin(t * 0.5) * 3.0, std::cos(t * 0.5) * 3.0, 0.0};
    }
    return batch;
}

std::vector<imu::Vec3> make_synthetic_vecs(std::size_t n) {
    std::vector<imu::Vec3> vs(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double t = static_cast<double>(i) * 0.01;
        vs[i] = {std::sin(t), std::cos(t), 1.0};
    }
    return vs;
}

}  // namespace

static void BM_ProcessBatch(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const imu::IMUSampleBatch raw = make_synthetic_batch(n);
    const imu::ProcessBatchParams params;
    for (auto _ : state) {
        auto out = imu::process_batch(raw, params);
        benchmark::DoNotOptimize(out);
    }
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * n));
}
BENCHMARK(BM_ProcessBatch)->Arg(1'000)->Arg(10'000)->Arg(100'000)->Arg(1'000'000);

static void BM_Vec3NormBatch_CPU(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const std::vector<imu::Vec3> vs = make_synthetic_vecs(n);
    for (auto _ : state) {
        auto out = imu::math::vec3_norm_batch(vs);
        benchmark::DoNotOptimize(out);
    }
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * n));
}
BENCHMARK(BM_Vec3NormBatch_CPU)->Arg(1'000)->Arg(100'000)->Arg(1'000'000)->Arg(10'000'000);

BENCHMARK_MAIN();

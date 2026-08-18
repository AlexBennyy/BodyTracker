// GPU counterpart to BM_Vec3NormBatch_CPU in cpu_benchmark.cpp -- the only
// piece of the pipeline currently ported to CUDA (see src/imu_cuda.cu).
#include <benchmark/benchmark.h>

#include <cmath>
#include <vector>

#include "imu_cuda.hpp"

namespace {

std::vector<double> make_synthetic_xyz(std::size_t n) {
    std::vector<double> xyz(3 * n);
    for (std::size_t i = 0; i < n; ++i) {
        const double t = static_cast<double>(i) * 0.01;
        xyz[3 * i] = std::sin(t);
        xyz[3 * i + 1] = std::cos(t);
        xyz[3 * i + 2] = 1.0;
    }
    return xyz;
}

}  // namespace

static void BM_Vec3NormBatch_CUDA(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const std::vector<double> xyz = make_synthetic_xyz(n);
    std::vector<double> out(n);
    for (auto _ : state) {
        imu::cuda::vec3_norm_batch(xyz.data(), n, out.data());
        benchmark::DoNotOptimize(out);
    }
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * n));
}
BENCHMARK(BM_Vec3NormBatch_CUDA)->Arg(1'000)->Arg(100'000)->Arg(1'000'000)->Arg(10'000'000);

BENCHMARK_MAIN();

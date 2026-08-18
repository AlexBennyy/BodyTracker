#include "imu_cuda.hpp"

#include <cstdio>
#include <cstdlib>

namespace {

#define CUDA_CHECK(expr)                                                             \
    do {                                                                             \
        cudaError_t _err = (expr);                                                   \
        if (_err != cudaSuccess) {                                                   \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                          cudaGetErrorString(_err));                                 \
            std::abort();                                                            \
        }                                                                            \
    } while (0)

__global__ void vec3_norm_kernel(const double* xyz, std::size_t n, double* out) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i < n) {
        const double x = xyz[3 * i];
        const double y = xyz[3 * i + 1];
        const double z = xyz[3 * i + 2];
        out[i] = sqrt(x * x + y * y + z * z);
    }
}

}  // namespace

namespace imu::cuda {

bool available() { return true; }

void vec3_norm_batch(const double* xyz, std::size_t n, double* out_norms) {
    if (n == 0) return;

    double* d_xyz = nullptr;
    double* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_xyz, 3 * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_xyz, xyz, 3 * n * sizeof(double), cudaMemcpyHostToDevice));

    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
    vec3_norm_kernel<<<blocks, kThreads>>>(d_xyz, n, d_out);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(out_norms, d_out, n * sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_xyz));
    CUDA_CHECK(cudaFree(d_out));
}

}  // namespace imu::cuda

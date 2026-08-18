#pragma once

#include <cstddef>

// Minimal CUDA-accelerated counterpart to imu::math::vec3_norm_batch
// (see include/imu_math.hpp). Only element-wise batch operations are CUDA
// candidates for now -- the bias/low-pass/complementary-filter pipeline in
// imu_processor is sequential in time per sensor and isn't a good fit for a
// first GPU port.
namespace imu::cuda {

// True if this build was compiled with CUDA support (BUILD_CUDA=ON and a
// working nvcc). Callers should check this before calling vec3_norm_batch.
bool available();

// xyz: n interleaved (x,y,z) triples (length 3*n). out_norms: length n.
// Handles host<->device transfer internally; not meant to be called in a
// tight loop over small batches.
void vec3_norm_batch(const double* xyz, std::size_t n, double* out_norms);

}  // namespace imu::cuda

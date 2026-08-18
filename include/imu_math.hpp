#pragma once

#include "imu_types.hpp"

// Core vector math and single-channel filters used by the IMU pipeline.
// Ported from python/imu_preprocess.py (vec_*, LowPass3) and
// python/imu_features.py (ComplementaryPitch, rad2deg/deg2rad).
namespace imu::math {

Vec3 vec_add(const Vec3& a, const Vec3& b);
Vec3 vec_sub(const Vec3& a, const Vec3& b);
Vec3 vec_scale(const Vec3& a, double s);
double vec_norm(const Vec3& a);

// Element-wise norm over a batch of vectors. Each output only depends on its
// own input, so unlike the sequential filters below this is embarrassingly
// parallel -- it's the CPU baseline the CUDA kernel in src/imu_cuda.cu is
// benchmarked against.
std::vector<double> vec3_norm_batch(const std::vector<Vec3>& vs);

constexpr double kRadToDeg = 180.0 / 3.14159265358979323846;
constexpr double kDegToRad = 3.14159265358979323846 / 180.0;

double rad2deg(double r);
double deg2rad(double d);

// 1st-order IIR low-pass filter applied independently to each axis of a Vec3.
class LowPass3 {
public:
    explicit LowPass3(double alpha) : alpha_(alpha) {}

    Vec3 update(const Vec3& x);
    void reset() { initialized_ = false; }

private:
    double alpha_;
    Vec3 y_{};
    bool initialized_ = false;
};

// Complementary filter estimating pitch (degrees) from accel + gyro.
class ComplementaryPitch {
public:
    ComplementaryPitch(double alpha, GyroUnits gyro_units)
        : alpha_(alpha), gyro_units_(gyro_units) {}

    // pitch_axis: which gyro axis (0=x,1=y,2=z) best tracks pitch rate for the mount.
    double update_pitch_deg(const Vec3& a, const Vec3& g, double dt, int pitch_axis);

    bool has_estimate() const { return has_pitch_; }
    double pitch_deg() const { return pitch_deg_; }

private:
    Vec3 gyro_to_deg_per_s(const Vec3& g) const;

    double alpha_;
    GyroUnits gyro_units_;
    double pitch_deg_ = 0.0;
    bool has_pitch_ = false;
};

}  // namespace imu::math

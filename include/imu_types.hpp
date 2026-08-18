#pragma once

#include <cstdint>
#include <vector>

namespace imu {

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

// One synchronized sample from the two-IMU rig (mirrors Python IMUFrame).
struct IMUSample {
    std::int64_t t_ms = 0;
    Vec3 a1;  // forearm accel
    Vec3 g1;  // forearm gyro
    Vec3 a2;  // upper-arm accel
    Vec3 g2;  // upper-arm gyro
};

// Per-axis mean accel/gyro captured while the rig is held still (mirrors Python Bias).
struct Bias {
    Vec3 a;
    Vec3 g;
};

enum class GyroUnits {
    Raw,
    Dps,
    Rads,
};

// One derived feature sample (mirrors Python FeatureFrame).
struct FeatureSample {
    std::int64_t t_ms = 0;
    double dt = 0.0;

    double pitch1 = 0.0;
    double pitch2 = 0.0;
    double elbow_flex = 0.0;

    double elbow_vel = 0.0;
    double g1_mag = 0.0;
    double g2_mag = 0.0;

    double upper_arm_motion = 0.0;
    double jerk1 = 0.0;
    double jerk2 = 0.0;
    double a1_mag = 0.0;
    double a2_mag = 0.0;
};

using IMUSampleBatch = std::vector<IMUSample>;
using FeatureSampleBatch = std::vector<FeatureSample>;

}  // namespace imu

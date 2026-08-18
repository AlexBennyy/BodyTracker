#include <gtest/gtest.h>

#include <cmath>

#include "imu_math.hpp"
#include "imu_processor.hpp"

using imu::Bias;
using imu::FeatureSampleBatch;
using imu::GyroUnits;
using imu::IMUSample;
using imu::IMUSampleBatch;
using imu::ProcessBatchParams;
using imu::Vec3;
namespace math = imu::math;

namespace {
constexpr double kEps = 1e-9;
}

TEST(VecMath, AddSubScaleNorm) {
    const Vec3 a{1.0, 2.0, 3.0};
    const Vec3 b{0.5, -1.0, 2.0};

    const Vec3 sum = math::vec_add(a, b);
    EXPECT_NEAR(sum.x, 1.5, kEps);
    EXPECT_NEAR(sum.y, 1.0, kEps);
    EXPECT_NEAR(sum.z, 5.0, kEps);

    const Vec3 diff = math::vec_sub(a, b);
    EXPECT_NEAR(diff.x, 0.5, kEps);
    EXPECT_NEAR(diff.y, 3.0, kEps);
    EXPECT_NEAR(diff.z, 1.0, kEps);

    const Vec3 scaled = math::vec_scale(a, 2.0);
    EXPECT_NEAR(scaled.x, 2.0, kEps);
    EXPECT_NEAR(scaled.y, 4.0, kEps);
    EXPECT_NEAR(scaled.z, 6.0, kEps);

    EXPECT_NEAR(math::vec_norm(Vec3{3.0, 4.0, 0.0}), 5.0, kEps);
}

TEST(VecMath, RadDegRoundTrip) {
    EXPECT_NEAR(math::rad2deg(M_PI), 180.0, 1e-9);
    EXPECT_NEAR(math::deg2rad(180.0), M_PI, 1e-9);
}

TEST(LowPass3Test, FirstSamplePassesThrough) {
    math::LowPass3 lp(0.2);
    const Vec3 out = lp.update(Vec3{1.0, 2.0, 3.0});
    EXPECT_NEAR(out.x, 1.0, kEps);
    EXPECT_NEAR(out.y, 2.0, kEps);
    EXPECT_NEAR(out.z, 3.0, kEps);
}

TEST(LowPass3Test, BlendsTowardNewSample) {
    math::LowPass3 lp(0.5);
    lp.update(Vec3{0.0, 0.0, 0.0});
    const Vec3 out = lp.update(Vec3{10.0, 0.0, 0.0});
    // alpha=0.5: y = 0.5*10 + 0.5*0
    EXPECT_NEAR(out.x, 5.0, kEps);
}

TEST(ComplementaryPitchTest, FirstSampleUsesAccelOnly) {
    math::ComplementaryPitch f(0.98, GyroUnits::Dps);
    // ax=0, ay=0, az=1 -> pitch_acc = atan2(-0, sqrt(0+1)) = 0 deg
    const double pitch = f.update_pitch_deg(Vec3{0.0, 0.0, 1.0}, Vec3{0.0, 0.0, 0.0}, 0.01, 1);
    EXPECT_NEAR(pitch, 0.0, 1e-6);
}

TEST(ComplementaryPitchTest, BlendsGyroIntegrationWithAccel) {
    math::ComplementaryPitch f(0.98, GyroUnits::Dps);
    f.update_pitch_deg(Vec3{0.0, 0.0, 1.0}, Vec3{0.0, 0.0, 0.0}, 0.01, 1);
    // second call: gyro rate on axis 1 = 2 dps, accel still gives pitch_acc = 0
    const double pitch = f.update_pitch_deg(Vec3{0.0, 0.0, 1.0}, Vec3{0.0, 2.0, 0.0}, 0.01, 1);
    // pitch_gyro = 0 + 2*0.01 = 0.02; pitch = 0.98*0.02 + 0.02*0 = 0.0196
    EXPECT_NEAR(pitch, 0.0196, 1e-6);
}

TEST(CalibrateBias, MeansEachAxis) {
    const std::vector<Vec3> accel = {{1.0, 2.0, 3.0}, {3.0, 4.0, 5.0}};
    const std::vector<Vec3> gyro = {{0.0, 0.0, 0.0}, {2.0, 2.0, 2.0}};
    const Bias bias = imu::calibrate_bias(accel, gyro);
    EXPECT_NEAR(bias.a.x, 2.0, kEps);
    EXPECT_NEAR(bias.a.y, 3.0, kEps);
    EXPECT_NEAR(bias.a.z, 4.0, kEps);
    EXPECT_NEAR(bias.g.x, 1.0, kEps);
    EXPECT_NEAR(bias.g.y, 1.0, kEps);
    EXPECT_NEAR(bias.g.z, 1.0, kEps);
}

TEST(CalibrateBias, EmptyBufferIsZero) {
    const Bias bias = imu::calibrate_bias({}, {});
    EXPECT_NEAR(bias.a.x, 0.0, kEps);
    EXPECT_NEAR(bias.g.z, 0.0, kEps);
}

TEST(ProcessBatch, EmptyAndSingleSampleEmitNothing) {
    ProcessBatchParams params;
    EXPECT_TRUE(imu::process_batch({}, params).empty());

    IMUSampleBatch one(1);
    one[0].t_ms = 0;
    one[0].a1 = {0.0, 0.0, 1.0};
    EXPECT_TRUE(imu::process_batch(one, params).empty())
        << "first sample only seeds filter state, matching FeatureEngine.update";
}

TEST(ProcessBatch, SkipsOutOfRangeDt) {
    ProcessBatchParams params;
    IMUSampleBatch raw(3);
    raw[0] = {0, {0, 0, 1}, {0, 0, 0}, {0, 0, 1}, {0, 0, 0}};
    raw[1] = {0, {0, 0, 1}, {0, 0, 0}, {0, 0, 1}, {0, 0, 0}};    // dt=0 -> skipped
    raw[2] = {1000, {0, 0, 1}, {0, 0, 0}, {0, 0, 1}, {0, 0, 0}};  // dt=1.0 > 0.2 -> skipped

    const FeatureSampleBatch out = imu::process_batch(raw, params);
    EXPECT_TRUE(out.empty());
}

TEST(ProcessBatch, SteadyStateMatchesHandComputedValues) {
    ProcessBatchParams params;  // defaults: alpha_accel=0.2, alpha_gyro=0.2, comp=0.98, Dps, axis=1
    IMUSampleBatch raw(2);
    raw[0] = {0, {0, 0, 1}, {0, 0, 0}, {0, 0, 1}, {0, 0, 0}};
    raw[1] = {10, {0, 0, 1}, {0, 0, 0}, {0, 0, 1}, {0, 0, 0}};

    const FeatureSampleBatch out = imu::process_batch(raw, params);
    ASSERT_EQ(out.size(), 1u);
    const auto& s = out[0];
    EXPECT_EQ(s.t_ms, 10);
    EXPECT_NEAR(s.dt, 0.01, 1e-9);
    EXPECT_NEAR(s.pitch1, 0.0, 1e-9);
    EXPECT_NEAR(s.pitch2, 0.0, 1e-9);
    EXPECT_NEAR(s.elbow_flex, 0.0, 1e-9);
    EXPECT_NEAR(s.elbow_vel, 0.0, 1e-9);
    EXPECT_NEAR(s.g1_mag, 0.0, 1e-9);
    EXPECT_NEAR(s.jerk1, 0.0, 1e-9);
    EXPECT_NEAR(s.a1_mag, 1.0, 1e-9);
    EXPECT_NEAR(s.a2_mag, 1.0, 1e-9);
}

TEST(ProcessBatch, GyroDrivenPitchAndElbowVelocity) {
    ProcessBatchParams params;
    IMUSampleBatch raw(3);
    raw[0] = {0, {0, 0, 1}, {0, 0, 0}, {0, 0, 1}, {0, 0, 0}};
    raw[1] = {10, {0, 0, 1}, {0, 0, 0}, {0, 0, 1}, {0, 0, 0}};
    raw[2] = {20, {0, 0, 1}, {0, 10, 0}, {0, 0, 1}, {0, 0, 0}};  // forearm gyro y=10 dps

    const FeatureSampleBatch out = imu::process_batch(raw, params);
    ASSERT_EQ(out.size(), 2u);
    const auto& s = out[1];
    // g1 low-pass: 0.2*(0,10,0) + 0.8*(0,0,0) = (0,2,0) -> gyro_rate=2 dps
    // pitch_gyro = 0 + 2*0.01 = 0.02; pitch1 = 0.98*0.02 + 0.02*0 = 0.0196
    EXPECT_NEAR(s.pitch1, 0.0196, 1e-6);
    EXPECT_NEAR(s.pitch2, 0.0, 1e-9);
    EXPECT_NEAR(s.elbow_flex, 0.0196, 1e-6);
    EXPECT_NEAR(s.elbow_vel, 1.96, 1e-4);  // (0.0196 - 0.0) / 0.01
    EXPECT_NEAR(s.g1_mag, 2.0, 1e-9);
}

TEST(ProcessBatch, BiasIsSubtractedBeforeFiltering) {
    ProcessBatchParams params;
    params.has_bias1 = true;
    params.bias1 = Bias{Vec3{0.0, 0.0, 1.0}, Vec3{0.0, 0.0, 0.0}};  // cancel out the 1g z-axis

    IMUSampleBatch raw(2);
    raw[0] = {0, {0, 0, 1}, {0, 0, 0}, {0, 0, 1}, {0, 0, 0}};
    raw[1] = {10, {0, 0, 1}, {0, 0, 0}, {0, 0, 1}, {0, 0, 0}};

    const FeatureSampleBatch out = imu::process_batch(raw, params);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_NEAR(out[0].a1_mag, 0.0, 1e-9);  // (0,0,1) - (0,0,1) bias = (0,0,0)
}

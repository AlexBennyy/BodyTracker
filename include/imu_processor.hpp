#pragma once

#include <vector>

#include "imu_types.hpp"

// Batch/offline equivalent of the realtime pipeline in python/imu_preprocess.py
// (IMUPreprocessor bias removal + low-pass) and python/imu_features.py
// (FeatureEngine complementary-filter pitch + derived motion features).
//
// Unlike the Python version this does not touch a serial port or a thread --
// it consumes an already-recorded batch of samples and returns the derived
// feature stream. That framing is what makes it a candidate for a CUDA
// implementation later: many independent recordings (or independent
// per-axis low-pass channels) can be processed in parallel, even though a
// single recording's filters are inherently sequential in time.
namespace imu {

struct ProcessBatchParams {
    double alpha_accel = 0.2;
    double alpha_gyro = 0.2;
    double complementary_alpha = 0.98;
    GyroUnits gyro_units = GyroUnits::Dps;
    int pitch_axis_imu1 = 1;  // 0=x,1=y,2=z
    int pitch_axis_imu2 = 1;

    bool has_bias1 = false;
    Bias bias1{};
    bool has_bias2 = false;
    Bias bias2{};
};

// Mean of paired accel/gyro samples captured while the rig is held still.
Bias calibrate_bias(const std::vector<Vec3>& accel_samples, const std::vector<Vec3>& gyro_samples);

// Runs bias removal + low-pass filtering + complementary pitch + motion
// features over `raw` in time order. `raw` must already be sorted by t_ms.
// The first sample only seeds filter state and emits no output, matching
// FeatureEngine.update's behavior on the first frame.
FeatureSampleBatch process_batch(const IMUSampleBatch& raw, const ProcessBatchParams& params);

}  // namespace imu

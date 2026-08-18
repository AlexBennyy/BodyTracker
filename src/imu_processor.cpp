#include "imu_processor.hpp"

#include <cmath>

#include "imu_math.hpp"

namespace imu {

namespace {

Vec3 mean3(const std::vector<Vec3>& buf) {
    if (buf.empty()) return Vec3{};
    double sx = 0.0, sy = 0.0, sz = 0.0;
    for (const auto& v : buf) {
        sx += v.x;
        sy += v.y;
        sz += v.z;
    }
    const double n = static_cast<double>(buf.size());
    return {sx / n, sy / n, sz / n};
}

}  // namespace

Bias calibrate_bias(const std::vector<Vec3>& accel_samples,
                     const std::vector<Vec3>& gyro_samples) {
    return Bias{mean3(accel_samples), mean3(gyro_samples)};
}

FeatureSampleBatch process_batch(const IMUSampleBatch& raw, const ProcessBatchParams& params) {
    FeatureSampleBatch out;
    if (raw.empty()) return out;
    out.reserve(raw.size());

    math::LowPass3 lp_a1(params.alpha_accel);
    math::LowPass3 lp_g1(params.alpha_gyro);
    math::LowPass3 lp_a2(params.alpha_accel);
    math::LowPass3 lp_g2(params.alpha_gyro);
    math::ComplementaryPitch f1(params.complementary_alpha, params.gyro_units);
    math::ComplementaryPitch f2(params.complementary_alpha, params.gyro_units);

    bool have_prev_t = false;
    std::int64_t prev_t_ms = 0;

    bool have_prev_elbow = false;
    double prev_elbow_flex = 0.0;

    bool have_prev_a = false;
    Vec3 prev_a1{};
    Vec3 prev_a2{};

    bool have_prev_pitch2 = false;
    double prev_pitch2 = 0.0;

    for (const auto& fr : raw) {
        Vec3 a1 = fr.a1;
        Vec3 g1 = fr.g1;
        Vec3 a2 = fr.a2;
        Vec3 g2 = fr.g2;

        if (params.has_bias1) {
            a1 = math::vec_sub(a1, params.bias1.a);
            g1 = math::vec_sub(g1, params.bias1.g);
        }
        if (params.has_bias2) {
            a2 = math::vec_sub(a2, params.bias2.a);
            g2 = math::vec_sub(g2, params.bias2.g);
        }

        const Vec3 a1f = lp_a1.update(a1);
        const Vec3 g1f = lp_g1.update(g1);
        const Vec3 a2f = lp_a2.update(a2);
        const Vec3 g2f = lp_g2.update(g2);

        if (!have_prev_t) {
            have_prev_t = true;
            prev_t_ms = fr.t_ms;
            prev_a1 = a1f;
            prev_a2 = a2f;
            have_prev_a = true;
            continue;  // seed only, no output -- matches FeatureEngine.update's first call
        }

        const double dt = static_cast<double>(fr.t_ms - prev_t_ms) / 1000.0;
        prev_t_ms = fr.t_ms;
        if (dt <= 0.0 || dt > 0.2) {
            continue;  // skip weird time jumps, same as the Python pipeline
        }

        const double pitch1 = f1.update_pitch_deg(a1f, g1f, dt, params.pitch_axis_imu1);
        const double pitch2 = f2.update_pitch_deg(a2f, g2f, dt, params.pitch_axis_imu2);
        const double elbow_flex = pitch1 - pitch2;

        double elbow_vel = 0.0;
        if (have_prev_elbow) {
            elbow_vel = (elbow_flex - prev_elbow_flex) / dt;
        }
        prev_elbow_flex = elbow_flex;
        have_prev_elbow = true;

        const double g1_mag = math::vec_norm(g1f);
        const double g2_mag = math::vec_norm(g2f);

        double upper_arm_motion = 0.0;
        if (have_prev_pitch2) {
            upper_arm_motion = std::abs(pitch2 - prev_pitch2) / dt;
        }
        prev_pitch2 = pitch2;
        have_prev_pitch2 = true;

        double jerk1 = 0.0;
        double jerk2 = 0.0;
        if (have_prev_a) {
            jerk1 = math::vec_norm(math::vec_scale(math::vec_sub(a1f, prev_a1), 1.0 / dt));
            jerk2 = math::vec_norm(math::vec_scale(math::vec_sub(a2f, prev_a2), 1.0 / dt));
        }
        prev_a1 = a1f;
        prev_a2 = a2f;

        FeatureSample sample;
        sample.t_ms = fr.t_ms;
        sample.dt = dt;
        sample.pitch1 = pitch1;
        sample.pitch2 = pitch2;
        sample.elbow_flex = elbow_flex;
        sample.elbow_vel = elbow_vel;
        sample.g1_mag = g1_mag;
        sample.g2_mag = g2_mag;
        sample.upper_arm_motion = upper_arm_motion;
        sample.jerk1 = jerk1;
        sample.jerk2 = jerk2;
        sample.a1_mag = math::vec_norm(a1f);
        sample.a2_mag = math::vec_norm(a2f);
        out.push_back(sample);
    }

    return out;
}

}  // namespace imu

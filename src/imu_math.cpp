#include "imu_math.hpp"

#include <cmath>

namespace imu::math {

Vec3 vec_add(const Vec3& a, const Vec3& b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 vec_sub(const Vec3& a, const Vec3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec3 vec_scale(const Vec3& a, double s) { return {a.x * s, a.y * s, a.z * s}; }

double vec_norm(const Vec3& a) { return std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z); }

std::vector<double> vec3_norm_batch(const std::vector<Vec3>& vs) {
    std::vector<double> out(vs.size());
    for (std::size_t i = 0; i < vs.size(); ++i) out[i] = vec_norm(vs[i]);
    return out;
}

double rad2deg(double r) { return r * kRadToDeg; }
double deg2rad(double d) { return d * kDegToRad; }

Vec3 LowPass3::update(const Vec3& x) {
    if (!initialized_) {
        y_ = x;
        initialized_ = true;
        return y_;
    }
    y_ = {
        alpha_ * x.x + (1.0 - alpha_) * y_.x,
        alpha_ * x.y + (1.0 - alpha_) * y_.y,
        alpha_ * x.z + (1.0 - alpha_) * y_.z,
    };
    return y_;
}

Vec3 ComplementaryPitch::gyro_to_deg_per_s(const Vec3& g) const {
    switch (gyro_units_) {
        case GyroUnits::Dps:
            return g;
        case GyroUnits::Rads:
            return {rad2deg(g.x), rad2deg(g.y), rad2deg(g.z)};
        case GyroUnits::Raw:
        default:
            // Unit unknown: caller should set Dps upstream once known. Pass through.
            return g;
    }
}

double ComplementaryPitch::update_pitch_deg(const Vec3& a, const Vec3& g, double dt,
                                             int pitch_axis) {
    const Vec3 g_dps = gyro_to_deg_per_s(g);

    const double pitch_acc = rad2deg(std::atan2(-a.x, std::sqrt(a.y * a.y + a.z * a.z)));

    const double gyro_rate = pitch_axis == 0 ? g_dps.x : pitch_axis == 1 ? g_dps.y : g_dps.z;

    if (!has_pitch_) {
        pitch_deg_ = pitch_acc;
        has_pitch_ = true;
    } else {
        const double pitch_gyro = pitch_deg_ + gyro_rate * dt;
        pitch_deg_ = alpha_ * pitch_gyro + (1.0 - alpha_) * pitch_acc;
    }
    return pitch_deg_;
}

}  // namespace imu::math

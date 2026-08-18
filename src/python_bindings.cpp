// pybind11 bindings exposing the C++ batch-processing core to Python as
// `_imu_cpp`. Inputs/outputs are numpy arrays so this slots in next to the
// existing numpy-based Python code (see python/imu_preprocess.py,
// python/imu_features.py) without changing how callers already think about
// the data.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>
#include <string>

#include "imu_math.hpp"
#include "imu_processor.hpp"

namespace py = pybind11;

namespace {

using ArrayD = py::array_t<double, py::array::c_style | py::array::forcecast>;
using ArrayI64 = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;

imu::Vec3 vec3_row(const ArrayD& arr, py::ssize_t i) {
    auto buf = arr.unchecked<2>();
    return {buf(i, 0), buf(i, 1), buf(i, 2)};
}

imu::GyroUnits parse_gyro_units(const std::string& s) {
    if (s == "dps") return imu::GyroUnits::Dps;
    if (s == "rads") return imu::GyroUnits::Rads;
    if (s == "raw") return imu::GyroUnits::Raw;
    throw std::invalid_argument("gyro_units must be one of 'raw', 'dps', 'rads'");
}

py::tuple vec3_to_tuple(const imu::Vec3& v) { return py::make_tuple(v.x, v.y, v.z); }

// bias: None, or ((ax, ay, az), (gx, gy, gz)) -- mirrors python imu_preprocess.Bias.
bool parse_bias(const py::object& obj, imu::Bias& out) {
    if (obj.is_none()) return false;
    py::tuple pair = obj.cast<py::tuple>();
    if (pair.size() != 2) throw std::invalid_argument("bias must be ((ax,ay,az), (gx,gy,gz))");
    py::tuple a = pair[0].cast<py::tuple>();
    py::tuple g = pair[1].cast<py::tuple>();
    out.a = {a[0].cast<double>(), a[1].cast<double>(), a[2].cast<double>()};
    out.g = {g[0].cast<double>(), g[1].cast<double>(), g[2].cast<double>()};
    return true;
}

py::tuple calibrate_bias_np(const ArrayD& accel, const ArrayD& gyro) {
    if (accel.ndim() != 2 || accel.shape(1) != 3 || gyro.ndim() != 2 || gyro.shape(1) != 3) {
        throw std::invalid_argument("accel/gyro must be shape (N, 3)");
    }
    std::vector<imu::Vec3> a(static_cast<size_t>(accel.shape(0)));
    for (py::ssize_t i = 0; i < accel.shape(0); ++i) a[static_cast<size_t>(i)] = vec3_row(accel, i);
    std::vector<imu::Vec3> g(static_cast<size_t>(gyro.shape(0)));
    for (py::ssize_t i = 0; i < gyro.shape(0); ++i) g[static_cast<size_t>(i)] = vec3_row(gyro, i);

    const imu::Bias bias = imu::calibrate_bias(a, g);
    return py::make_tuple(vec3_to_tuple(bias.a), vec3_to_tuple(bias.g));
}

py::dict process_batch_np(const ArrayI64& t_ms, const ArrayD& a1, const ArrayD& g1,
                           const ArrayD& a2, const ArrayD& g2, double alpha_accel,
                           double alpha_gyro, double complementary_alpha,
                           const std::string& gyro_units, int pitch_axis_imu1,
                           int pitch_axis_imu2, py::object bias1, py::object bias2) {
    const py::ssize_t n = t_ms.shape(0);
    if (a1.shape(0) != n || g1.shape(0) != n || a2.shape(0) != n || g2.shape(0) != n) {
        throw std::invalid_argument("t_ms/a1/g1/a2/g2 must all have the same length");
    }
    if (a1.ndim() != 2 || a1.shape(1) != 3 || g1.ndim() != 2 || g1.shape(1) != 3 ||
        a2.ndim() != 2 || a2.shape(1) != 3 || g2.ndim() != 2 || g2.shape(1) != 3) {
        throw std::invalid_argument("a1/g1/a2/g2 must be shape (N, 3)");
    }

    imu::IMUSampleBatch raw(static_cast<size_t>(n));
    auto t_buf = t_ms.unchecked<1>();
    for (py::ssize_t i = 0; i < n; ++i) {
        auto& fr = raw[static_cast<size_t>(i)];
        fr.t_ms = t_buf(i);
        fr.a1 = vec3_row(a1, i);
        fr.g1 = vec3_row(g1, i);
        fr.a2 = vec3_row(a2, i);
        fr.g2 = vec3_row(g2, i);
    }

    imu::ProcessBatchParams params;
    params.alpha_accel = alpha_accel;
    params.alpha_gyro = alpha_gyro;
    params.complementary_alpha = complementary_alpha;
    params.gyro_units = parse_gyro_units(gyro_units);
    params.pitch_axis_imu1 = pitch_axis_imu1;
    params.pitch_axis_imu2 = pitch_axis_imu2;
    params.has_bias1 = parse_bias(bias1, params.bias1);
    params.has_bias2 = parse_bias(bias2, params.bias2);

    const imu::FeatureSampleBatch out = imu::process_batch(raw, params);
    const py::ssize_t m = static_cast<py::ssize_t>(out.size());

    ArrayI64 out_t_ms(m);
    ArrayD out_dt(m), out_pitch1(m), out_pitch2(m), out_elbow_flex(m), out_elbow_vel(m),
        out_g1_mag(m), out_g2_mag(m), out_upper_arm_motion(m), out_jerk1(m), out_jerk2(m),
        out_a1_mag(m), out_a2_mag(m);

    auto t_out = out_t_ms.mutable_unchecked<1>();
    auto dt_out = out_dt.mutable_unchecked<1>();
    auto pitch1_out = out_pitch1.mutable_unchecked<1>();
    auto pitch2_out = out_pitch2.mutable_unchecked<1>();
    auto elbow_flex_out = out_elbow_flex.mutable_unchecked<1>();
    auto elbow_vel_out = out_elbow_vel.mutable_unchecked<1>();
    auto g1_mag_out = out_g1_mag.mutable_unchecked<1>();
    auto g2_mag_out = out_g2_mag.mutable_unchecked<1>();
    auto upper_arm_motion_out = out_upper_arm_motion.mutable_unchecked<1>();
    auto jerk1_out = out_jerk1.mutable_unchecked<1>();
    auto jerk2_out = out_jerk2.mutable_unchecked<1>();
    auto a1_mag_out = out_a1_mag.mutable_unchecked<1>();
    auto a2_mag_out = out_a2_mag.mutable_unchecked<1>();

    for (py::ssize_t i = 0; i < m; ++i) {
        const auto& s = out[static_cast<size_t>(i)];
        t_out(i) = s.t_ms;
        dt_out(i) = s.dt;
        pitch1_out(i) = s.pitch1;
        pitch2_out(i) = s.pitch2;
        elbow_flex_out(i) = s.elbow_flex;
        elbow_vel_out(i) = s.elbow_vel;
        g1_mag_out(i) = s.g1_mag;
        g2_mag_out(i) = s.g2_mag;
        upper_arm_motion_out(i) = s.upper_arm_motion;
        jerk1_out(i) = s.jerk1;
        jerk2_out(i) = s.jerk2;
        a1_mag_out(i) = s.a1_mag;
        a2_mag_out(i) = s.a2_mag;
    }

    py::dict result;
    result["t_ms"] = out_t_ms;
    result["dt"] = out_dt;
    result["pitch1"] = out_pitch1;
    result["pitch2"] = out_pitch2;
    result["elbow_flex"] = out_elbow_flex;
    result["elbow_vel"] = out_elbow_vel;
    result["g1_mag"] = out_g1_mag;
    result["g2_mag"] = out_g2_mag;
    result["upper_arm_motion"] = out_upper_arm_motion;
    result["jerk1"] = out_jerk1;
    result["jerk2"] = out_jerk2;
    result["a1_mag"] = out_a1_mag;
    result["a2_mag"] = out_a2_mag;
    return result;
}

}  // namespace

PYBIND11_MODULE(_imu_cpp, m) {
    m.doc() = "C++ core for the dual-IMU body tracking pipeline (bias calibration + "
              "batch feature extraction). A CUDA implementation of the same batch "
              "operation is planned for benchmarking.";

    m.def("calibrate_bias", &calibrate_bias_np, py::arg("accel"), py::arg("gyro"),
          "Mean of (N,3) accel/gyro arrays captured while still. "
          "Returns ((ax,ay,az), (gx,gy,gz)).");

    m.def("process_batch", &process_batch_np, py::arg("t_ms"), py::arg("a1"), py::arg("g1"),
          py::arg("a2"), py::arg("g2"), py::arg("alpha_accel") = 0.2,
          py::arg("alpha_gyro") = 0.2, py::arg("complementary_alpha") = 0.98,
          py::arg("gyro_units") = "dps", py::arg("pitch_axis_imu1") = 1,
          py::arg("pitch_axis_imu2") = 1, py::arg("bias1") = py::none(),
          py::arg("bias2") = py::none(),
          "Batch bias-removal + low-pass + complementary-pitch feature extraction over "
          "a recorded IMU session. t_ms is (N,) int64; a1/g1/a2/g2 are (N,3) float64. "
          "Returns a dict of numpy arrays (t_ms, dt, pitch1, pitch2, elbow_flex, elbow_vel, "
          "g1_mag, g2_mag, upper_arm_motion, jerk1, jerk2, a1_mag, a2_mag) of length <= N.");
}

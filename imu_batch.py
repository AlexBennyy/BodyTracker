# imu_batch.py
"""
Offline/batch IMU processing: bias removal + low-pass filtering +
complementary-filter feature extraction over a full recorded session at once.

This is the batch counterpart to IMUPreprocessor (live, sample-by-sample) +
FeatureEngine (live, sample-by-sample) in imu_preprocess.py / imu_features.py
-- identical math, but processes a whole array of samples in one call so it
can use the compiled C++ library (via the _imu_cpp pybind11 extension, built
by CMake) when available, transparently falling back to a pure-Python
implementation of the same algorithm if the extension hasn't been built.

Input: t_ms (N,) int, a1/g1/a2/g2 (N,3) float -- same shape as mpu_dual_log.csv.
Output: dict of numpy arrays keyed by feature name (t_ms, dt, pitch1, pitch2,
elbow_flex, elbow_vel, g1_mag, g2_mag, upper_arm_motion, jerk1, jerk2,
a1_mag, a2_mag), one row per emitted feature sample (length <= N).
"""
from __future__ import annotations

import numpy as np

from imu_preprocess import LowPass3, IMUFrame, vec_sub
from imu_features import FeatureEngine

try:
    import _imu_cpp
    _HAVE_CPP = True
except ImportError:
    _HAVE_CPP = False

_FIELDS = ["t_ms", "dt", "pitch1", "pitch2", "elbow_flex", "elbow_vel",
           "g1_mag", "g2_mag", "upper_arm_motion", "jerk1", "jerk2",
           "a1_mag", "a2_mag"]


def using_cpp_backend() -> bool:
    """True if the compiled _imu_cpp extension is being used, False if the
    pure-Python fallback is active (e.g. the CMake project hasn't been built
    on this machine)."""
    return _HAVE_CPP


def calibrate_bias(accel, gyro):
    """Mean of (N,3) accel/gyro arrays captured while still.
    Returns ((ax,ay,az), (gx,gy,gz)). Same result via either backend."""
    if _HAVE_CPP:
        return _imu_cpp.calibrate_bias(
            np.asarray(accel, dtype=np.float64), np.asarray(gyro, dtype=np.float64)
        )
    accel = np.asarray(accel, dtype=np.float64)
    gyro = np.asarray(gyro, dtype=np.float64)
    if len(accel) == 0:
        a = (0.0, 0.0, 0.0)
    else:
        a = tuple(accel.mean(axis=0))
    if len(gyro) == 0:
        g = (0.0, 0.0, 0.0)
    else:
        g = tuple(gyro.mean(axis=0))
    return (a, g)


def process_batch(
    t_ms, a1, g1, a2, g2,
    alpha_accel=0.2, alpha_gyro=0.2, complementary_alpha=0.98,
    gyro_units="dps", pitch_axis_imu1=1, pitch_axis_imu2=1,
    bias1=None, bias2=None,
):
    """Run the bias/low-pass/complementary-filter pipeline over a full batch.

    Uses the compiled C++ extension when available (same math, much faster
    for large recordings -- see benchmarks/cpu_benchmark.cpp); otherwise
    falls back to a pure-Python implementation built from the same
    IMUPreprocessor/FeatureEngine classes used by the live pipeline, so
    results are identical either way (see the parity check that motivated
    this module).
    """
    if _HAVE_CPP:
        return _imu_cpp.process_batch(
            np.asarray(t_ms, dtype=np.int64),
            np.asarray(a1, dtype=np.float64),
            np.asarray(g1, dtype=np.float64),
            np.asarray(a2, dtype=np.float64),
            np.asarray(g2, dtype=np.float64),
            alpha_accel=alpha_accel, alpha_gyro=alpha_gyro,
            complementary_alpha=complementary_alpha, gyro_units=gyro_units,
            pitch_axis_imu1=pitch_axis_imu1, pitch_axis_imu2=pitch_axis_imu2,
            bias1=bias1, bias2=bias2,
        )
    return _process_batch_python(
        t_ms, a1, g1, a2, g2, alpha_accel, alpha_gyro, complementary_alpha,
        gyro_units, pitch_axis_imu1, pitch_axis_imu2, bias1, bias2,
    )


def _process_batch_python(
    t_ms, a1, g1, a2, g2, alpha_accel, alpha_gyro, complementary_alpha,
    gyro_units, pitch_axis_imu1, pitch_axis_imu2, bias1, bias2,
):
    n = len(t_ms)
    lp_a1 = LowPass3(alpha_accel)
    lp_g1 = LowPass3(alpha_gyro)
    lp_a2 = LowPass3(alpha_accel)
    lp_g2 = LowPass3(alpha_gyro)
    fe = FeatureEngine(alpha=complementary_alpha, gyro_units=gyro_units,
                        pitch_axis_imu1=pitch_axis_imu1, pitch_axis_imu2=pitch_axis_imu2)

    out = {f: [] for f in _FIELDS}

    for i in range(n):
        a1i, g1i = tuple(a1[i]), tuple(g1[i])
        a2i, g2i = tuple(a2[i]), tuple(g2[i])
        if bias1 is not None:
            a1i = vec_sub(a1i, bias1[0])
            g1i = vec_sub(g1i, bias1[1])
        if bias2 is not None:
            a2i = vec_sub(a2i, bias2[0])
            g2i = vec_sub(g2i, bias2[1])

        a1f = lp_a1.update(a1i)
        g1f = lp_g1.update(g1i)
        a2f = lp_a2.update(a2i)
        g2f = lp_g2.update(g2i)

        clean = IMUFrame(t_ms=int(t_ms[i]), a1=a1f, g1=g1f, a2=a2f, g2=g2f)
        ff = fe.update(clean)
        if ff is None:
            continue
        out["t_ms"].append(ff.t_ms)
        out["dt"].append(ff.dt)
        out["pitch1"].append(ff.pitch1)
        out["pitch2"].append(ff.pitch2)
        out["elbow_flex"].append(ff.elbow_flex)
        out["elbow_vel"].append(ff.elbow_vel)
        out["g1_mag"].append(ff.g1_mag)
        out["g2_mag"].append(ff.g2_mag)
        out["upper_arm_motion"].append(ff.upper_arm_motion)
        out["jerk1"].append(ff.jerk1)
        out["jerk2"].append(ff.jerk2)
        out["a1_mag"].append(ff.a1_mag)
        out["a2_mag"].append(ff.a2_mag)

    dtype_map = {"t_ms": np.int64}
    return {f: np.array(v, dtype=dtype_map.get(f, np.float64)) for f, v in out.items()}

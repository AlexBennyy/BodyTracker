# tests/test_parity.py
"""
Regression check: imu_batch.process_batch (C++ backend, when _imu_cpp is
built) must match the pure-Python fallback path -- which in turn is built
from the same IMUPreprocessor/FeatureEngine classes as the live pipeline --
to within floating-point noise, over a real recorded session.

Run directly: python3 tests/test_parity.py
If _imu_cpp hasn't been built (no CMake build in this checkout), the C++
vs Python comparison is skipped and only the Python fallback path itself is
sanity-checked, since there is nothing to compare it against.
"""
import csv
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import imu_batch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(REPO_ROOT, "mpu_dual_log.csv")


def _load_csv(path):
    rows = []
    with open(path) as f:
        r = csv.reader(f)
        next(r)  # header
        for parts in r:
            rows.append(list(map(float, parts)))
    t_ms = np.array([int(v[0]) for v in rows], dtype=np.int64)
    a1 = np.array([v[1:4] for v in rows])
    g1 = np.array([v[4:7] for v in rows])
    a2 = np.array([v[7:10] for v in rows])
    g2 = np.array([v[10:13] for v in rows])
    return t_ms, a1, g1, a2, g2


class ParityTest(unittest.TestCase):
    def test_cpp_matches_python_fallback(self):
        if not imu_batch.using_cpp_backend():
            self.skipTest("_imu_cpp extension not built in this checkout -- run cmake --build build")

        t_ms, a1, g1, a2, g2 = _load_csv(CSV_PATH)

        out_cpp = imu_batch.process_batch(t_ms, a1, g1, a2, g2, gyro_units="raw")
        out_py = imu_batch._process_batch_python(
            t_ms, a1, g1, a2, g2, 0.2, 0.2, 0.98, "raw", 1, 1, None, None
        )

        self.assertEqual(len(out_cpp["t_ms"]), len(out_py["t_ms"]))
        for field in imu_batch._FIELDS:
            diff = float(np.max(np.abs(out_cpp[field] - out_py[field])))
            self.assertLess(diff, 1e-8, f"field {field!r} diverged by {diff:.3e}")

    def test_python_fallback_runs(self):
        t_ms, a1, g1, a2, g2 = _load_csv(CSV_PATH)
        out = imu_batch._process_batch_python(
            t_ms, a1, g1, a2, g2, 0.2, 0.2, 0.98, "raw", 1, 1, None, None
        )
        self.assertGreater(len(out["t_ms"]), 0)
        self.assertEqual(len(out["t_ms"]), len(out["pitch1"]))


if __name__ == "__main__":
    unittest.main()

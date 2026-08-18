# imu_preprocess.py
from __future__ import annotations
from dataclasses import dataclass
import time
import threading
import serial
import math
from collections import deque
from typing import Optional, Tuple

# ---------- helpers -----------
def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

@dataclass
class IMUFrame:
    t_ms: int
    a1: Tuple[float, float, float]   # (ax, ay, az)
    g1: Tuple[float, float, float]   # (gx, gy, gz)
    a2: Tuple[float, float, float]
    g2: Tuple[float, float, float]

@dataclass
class Bias:
    a: Tuple[float, float, float]
    g: Tuple[float, float, float]

class LowPass3:
    """1st-order IIR low-pass per axis."""
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.y = (0.0, 0.0, 0.0)
        self.init = False

    def update(self, x: Tuple[float, float, float]) -> Tuple[float, float, float]:
        if not self.init:
            self.y = x
            self.init = True
            return self.y
        a = self.alpha
        self.y = (
            a * x[0] + (1 - a) * self.y[0],
            a * x[1] + (1 - a) * self.y[1],
            a * x[2] + (1 - a) * self.y[2],
        )
        return self.y

def vec_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

def vec_add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

def vec_scale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)

def vec_norm(a):
    return math.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])

# ---------- main pipeline ----------

class IMUPreprocessor:
    """
    Step 2: serial ingest + parse + bias calibration + low-pass filtering.

    Output frames are "clean": biases removed + filtered.
    """
    def __init__(
        self,
        port: str,
        baud: int = 115200,
        accel_units: str = "raw",  # "raw" or "g" or "mps2"
        gyro_units: str = "raw",   # "raw" or "dps" or "rads"
        # Low-pass smoothing: alpha in (0,1). Higher = less smoothing.
        alpha_accel: float = 0.2,
        alpha_gyro: float = 0.2,
        timeout: float = 1.0,
    ):
        self.ser = serial.Serial(port, baud, timeout=timeout)
        time.sleep(2)

        self.accel_units = accel_units
        self.gyro_units = gyro_units

        self.lp_a1 = LowPass3(alpha_accel)
        self.lp_g1 = LowPass3(alpha_gyro)
        self.lp_a2 = LowPass3(alpha_accel)
        self.lp_g2 = LowPass3(alpha_gyro)

        self.bias1: Optional[Bias] = None
        self.bias2: Optional[Bias] = None

        self.latest_raw: Optional[IMUFrame] = None
        self.latest_clean: Optional[IMUFrame] = None

        self._stop = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def close(self):
        self._stop = True
        try:
            self.ser.close()
        except:
            pass

    # ---- unit conversion (optional) ----
    def _convert_accel(self, a_raw: Tuple[float, float, float]) -> Tuple[float, float, float]:
        # If you already transmit in g or m/s^2, set accel_units accordingly.
        # For MPU6050 typical: raw -> g depends on accel full-scale.
        # Common setting ±2g: 16384 LSB/g.
        if self.accel_units == "raw":
            return a_raw
        lsb_per_g = 16384.0  # adjust if your accel range differs
        ax, ay, az = (a_raw[0] / lsb_per_g, a_raw[1] / lsb_per_g, a_raw[2] / lsb_per_g)
        if self.accel_units == "g":
            return (ax, ay, az)
        if self.accel_units == "mps2":
            g0 = 9.80665
            return (ax * g0, ay * g0, az * g0)
        return a_raw
    

    
    def _convert_gyro(self, g_raw: Tuple[float, float, float]) -> Tuple[float, float, float]:
        # Common MPU6050 setting ±250 dps: 131 LSB/(deg/s)
        if self.gyro_units == "raw":
            return g_raw
        lsb_per_dps = 131.0  # adjust if gyro range differs
        gx, gy, gz = (g_raw[0] / lsb_per_dps, g_raw[1] / lsb_per_dps, g_raw[2] / lsb_per_dps)
        if self.gyro_units == "dps":
            return (gx, gy, gz)
        if self.gyro_units == "rads":
            return (math.radians(gx), math.radians(gy), math.radians(gz))
        return g_raw

    # ---- parsing ----
    def _parse_line(self, line: str) -> Optional[IMUFrame]:
        line = line.strip()
        if not line:
            return None
        if line.startswith("t_ms") or line.startswith("ERROR"):
            return None

        parts = line.split(",")
        # tolerate a trailing comma
        parts = [p for p in parts if p != ""]
        if len(parts) < 13:
            return None

        try:
            vals = list(map(float, parts[:13]))
        except ValueError:
            return None

        t_ms = int(vals[0])

        a1_raw = (vals[1], vals[2], vals[3])
        g1_raw = (vals[4], vals[5], vals[6])
        a2_raw = (vals[7], vals[8], vals[9])
        g2_raw = (vals[10], vals[11], vals[12])

        a1 = self._convert_accel(a1_raw)
        g1 = self._convert_gyro(g1_raw)
        a2 = self._convert_accel(a2_raw)
        g2 = self._convert_gyro(g2_raw)

        return IMUFrame(t_ms=t_ms, a1=a1, g1=g1, a2=a2, g2=g2)

    def _apply_bias_and_filter(self, fr: IMUFrame) -> IMUFrame:
        a1, g1, a2, g2 = fr.a1, fr.g1, fr.a2, fr.g2

        if self.bias1 is not None:
            a1 = vec_sub(a1, self.bias1.a)
            g1 = vec_sub(g1, self.bias1.g)
        if self.bias2 is not None:
            a2 = vec_sub(a2, self.bias2.a)
            g2 = vec_sub(g2, self.bias2.g)

        # low-pass
        a1f = self.lp_a1.update(a1)
        g1f = self.lp_g1.update(g1)
        a2f = self.lp_a2.update(a2)
        g2f = self.lp_g2.update(g2)

        return IMUFrame(t_ms=fr.t_ms, a1=a1f, g1=g1f, a2=a2f, g2=g2f)

    def _loop(self):
        while not self._stop:
            try:
                line = self.ser.readline().decode("utf-8", errors="ignore")
            except Exception:
                continue

            fr = self._parse_line(line)
            if fr is None:
                continue

            clean = self._apply_bias_and_filter(fr)
            with self._lock:
                self.latest_raw = fr
                self.latest_clean = clean

    # ---- public API ----
    def get_latest(self) -> Optional[IMUFrame]:
        with self._lock:
            return self.latest_clean

    def calibrate_still(self, seconds: float = 2.0, max_samples: int = 400) -> None:
        """
        Tell user: stand still in neutral pose for ~2s.
        Computes gyro bias (and accel bias as mean; optional).
        """
        buf1_a, buf1_g = [], []
        buf2_a, buf2_g = [], []

        start = time.time()
        while time.time() - start < seconds and len(buf1_g) < max_samples:
            fr = None
            with self._lock:
                fr = self.latest_raw
            if fr is None:
                continue
            buf1_a.append(fr.a1); buf1_g.append(fr.g1)
            buf2_a.append(fr.a2); buf2_g.append(fr.g2)
            time.sleep(0.005)

        def mean3(buf):
            if not buf:
                return (0.0, 0.0, 0.0)
            sx = sum(v[0] for v in buf)
            sy = sum(v[1] for v in buf)
            sz = sum(v[2] for v in buf)
            n = float(len(buf))
            return (sx/n, sy/n, sz/n)

        # Gyro bias is the most important (should be ~0 when still)
        b1g = mean3(buf1_g)
        b2g = mean3(buf2_g)

        # Accel bias: for many metrics you can leave accel as-is.
        # If you want, subtract mean accel in neutral; this "zeros" gravity component in that pose.
        b1a = mean3(buf1_a)
        b2a = mean3(buf2_a)

        self.bias1 = Bias(a=b1a, g=b1g)
        self.bias2 = Bias(a=b2a, g=b2g)

        # Reset filters so they don't lag old data
        self.lp_a1.init = self.lp_g1.init = False
        self.lp_a2.init = self.lp_g2.init = False

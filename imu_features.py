# imu_features.py
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Optional, Tuple

from imu_preprocess import IMUFrame  # from Step 2

def vec_norm3(v: Tuple[float, float, float]) -> float:
    return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])

def safe_atan2(y, x) -> float:
    # returns radians
    return math.atan2(y, x)

def rad2deg(r: float) -> float:
    return r * (180.0 / math.pi)

def deg2rad(d: float) -> float:
    return d * (math.pi / 180.0)


@dataclass
class FeatureFrame:
    t_ms: int
    dt: float  # seconds since last

    # orientation-ish
    pitch1: float  # forearm pitch (deg)
    pitch2: float  # upper-arm pitch (deg)
    elbow_flex: float  # pitch1 - pitch2 (deg)

    # derivatives / motion
    elbow_vel: float  # deg/s
    g1_mag: float     # gyro magnitude forearm (unit depends on input)
    g2_mag: float     # gyro magnitude upper arm

    # "cheating"/control proxies
    upper_arm_motion: float  # abs change in pitch2 (deg/s approx)
    jerk1: float             # jerk magnitude forearm (units depend)
    jerk2: float             # jerk magnitude upper arm
    a1_mag: float
    a2_mag: float


class ComplementaryPitch:
    """
    Complementary filter for pitch only.
    - Uses gyro integration + accel-derived pitch correction.
    - Assumes gyro is in deg/s or rad/s depending on gyro_units.
    """
    def __init__(self, alpha: float = 0.98, gyro_units: str = "raw_or_unknown"):
        self.alpha = alpha
        self.pitch_deg: Optional[float] = None
        self.gyro_units = gyro_units  # "dps", "rads", or "raw_or_unknown"

    def _gyro_to_deg_per_s(self, gx, gy, gz) -> Tuple[float, float, float]:
        # We only use the gyro axis that best corresponds to "pitch rate".
        # BUT since mounting varies, we’ll approximate pitch rate using a single axis later.
        # This function just normalizes unit assumptions.
        if self.gyro_units == "dps":
            return (gx, gy, gz)
        if self.gyro_units == "rads":
            return (rad2deg(gx), rad2deg(gy), rad2deg(gz))
        # If unknown/raw: you should set gyro_units="dps" in Step 2 when you can.
        return (gx, gy, gz)

    def update_pitch_deg(
        self,
        a: Tuple[float, float, float],
        g: Tuple[float, float, float],
        dt: float,
        pitch_axis: int = 1,
    ) -> float:
        """
        a = (ax,ay,az) (any units; direction matters)
        g = (gx,gy,gz) (deg/s recommended)
        pitch_axis: which gyro axis corresponds most to pitch rate for your mount.
                   0=x,1=y,2=z. You'll pick the one that moves most during a curl.
                   Use a quck test (print raw gyro values while curling) to decide which axis to use for pitch correction.
        """
        ax, ay, az = a
        gx, gy, gz = self._gyro_to_deg_per_s(*g)

        # accel-derived pitch (deg)
        # common pitch formula (depends on axis convention):
        # pitch = atan2(-ax, sqrt(ay^2 + az^2))
        pitch_acc = rad2deg(safe_atan2(-ax, math.sqrt(ay*ay + az*az)))

        # gyro integration
        gyro_rate = (gx, gy, gz)[pitch_axis]
        if self.pitch_deg is None:
            self.pitch_deg = pitch_acc
        else:
            pitch_gyro = self.pitch_deg + gyro_rate * dt
            self.pitch_deg = self.alpha * pitch_gyro + (1.0 - self.alpha) * pitch_acc

        return self.pitch_deg


class FeatureEngine:
    """
    Step 3:
    - Estimate pitch for both IMUs (complementary filter)
    - Produce elbow_flex proxy and motion/quality features
    """
    def __init__(
        self,
        alpha: float = 0.98,
        gyro_units: str = "dps",   # set this to match Step 2 conversions
        pitch_axis_imu1: int = 1,  # choose after a quick test (see notes below)
        pitch_axis_imu2: int = 1,
    ):
        self.f1 = ComplementaryPitch(alpha=alpha, gyro_units=gyro_units)
        self.f2 = ComplementaryPitch(alpha=alpha, gyro_units=gyro_units)

        self.pitch_axis_imu1 = pitch_axis_imu1
        self.pitch_axis_imu2 = pitch_axis_imu2

        self._prev_t_ms: Optional[int] = None
        self._prev_elbow_flex: Optional[float] = None
        self._prev_a1: Optional[Tuple[float, float, float]] = None
        self._prev_a2: Optional[Tuple[float, float, float]] = None
        self._prev_pitch2: Optional[float] = None

    def update(self, fr: IMUFrame) -> Optional[FeatureFrame]:
        # dt
        if self._prev_t_ms is None:
            self._prev_t_ms = fr.t_ms
            self._prev_a1, self._prev_a2 = fr.a1, fr.a2
            return None

        dt = (fr.t_ms - self._prev_t_ms) / 1000.0
        self._prev_t_ms = fr.t_ms
        if dt <= 0 or dt > 0.2:
            # skip weird time jumps
            return None

        # pitch estimates (deg)
        pitch1 = self.f1.update_pitch_deg(fr.a1, fr.g1, dt, pitch_axis=self.pitch_axis_imu1)
        pitch2 = self.f2.update_pitch_deg(fr.a2, fr.g2, dt, pitch_axis=self.pitch_axis_imu2)

        elbow_flex = pitch1 - pitch2

        # elbow velocity (deg/s)
        elbow_vel = 0.0
        if self._prev_elbow_flex is not None:
            elbow_vel = (elbow_flex - self._prev_elbow_flex) / dt
        self._prev_elbow_flex = elbow_flex

        # gyro magnitudes
        g1_mag = vec_norm3(fr.g1)
        g2_mag = vec_norm3(fr.g2)

        # upper-arm motion proxy (deg/s approx using pitch2 change)
        upper_arm_motion = 0.0
        if self._prev_pitch2 is not None:
            upper_arm_motion = abs(pitch2 - self._prev_pitch2) / dt
        self._prev_pitch2 = pitch2

        # jerk magnitude from accel derivative (units depend on accel units)
        jerk1 = 0.0
        jerk2 = 0.0
        if self._prev_a1 is not None:
            da1 = ((fr.a1[0] - self._prev_a1[0]) / dt,
                   (fr.a1[1] - self._prev_a1[1]) / dt,
                   (fr.a1[2] - self._prev_a1[2]) / dt)
            jerk1 = vec_norm3(da1)
        if self._prev_a2 is not None:
            da2 = ((fr.a2[0] - self._prev_a2[0]) / dt,
                   (fr.a2[1] - self._prev_a2[1]) / dt,
                   (fr.a2[2] - self._prev_a2[2]) / dt)
            jerk2 = vec_norm3(da2)

        self._prev_a1, self._prev_a2 = fr.a1, fr.a2

        a1_mag = vec_norm3(fr.a1)
        a2_mag = vec_norm3(fr.a2)

        return FeatureFrame(
            t_ms=fr.t_ms, dt=dt,
            pitch1=pitch1, pitch2=pitch2, elbow_flex=elbow_flex,
            elbow_vel=elbow_vel,
            g1_mag=g1_mag, g2_mag=g2_mag,
            upper_arm_motion=upper_arm_motion,
            jerk1=jerk1, jerk2=jerk2,
            a1_mag=a1_mag, a2_mag=a2_mag,
        )

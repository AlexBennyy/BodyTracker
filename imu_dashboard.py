# imu_dashboard_csv.py
import argparse
import time
from dataclasses import dataclass
from collections import deque

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from imu_features import FeatureEngine, FeatureFrame


# -----------------------------
# 1) A "frame" that matches what FeatureEngine.update(fr) expects
# -----------------------------
# IMPORTANT: adjust field names to match your imu_features.FeatureEngine expectations.
# Common pattern: fr has imu1 + imu2 accel/gyro in consistent units.
@dataclass
class IMUFrame2:
    t_s: float

    # IMU1 accel (m/s^2) and gyro (deg/s)
    ax1: float; ay1: float; az1: float
    gx1: float; gy1: float; gz1: float

    # IMU2 accel (m/s^2) and gyro (deg/s)
    ax2: float; ay2: float; az2: float
    gx2: float; gy2: float; gz2: float


# -----------------------------
# 2) Your existing LiveDashboard (unchanged)
# -----------------------------
class LiveDashboard:
    def __init__(self, seconds=10.0, sample_hz=100):
        self.maxlen = int(seconds * sample_hz)
        self.t = deque(maxlen=self.maxlen)
        self.buf = {name: deque(maxlen=self.maxlen) for name in [
            "pitch1", "pitch2", "elbow_flex", "elbow_vel",
            "g1_mag", "g2_mag",
            "upper_arm_motion",
            "jerk1", "jerk2",
            "a1_mag", "a2_mag",
        ]}
        self.fig = {}
        self.ax = {}
        self.line = {}

        plt.ion()
        for name in self.buf.keys():
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title(name)
            ax.set_xlabel("time (s)")
            ax.set_ylabel(name)
            line, = ax.plot([], [])
            self.fig[name] = fig
            self.ax[name] = ax
            self.line[name] = line

        plt.show(block=False)

    def add(self, t_s: float, ff: FeatureFrame):
        self.t.append(t_s)
        self.buf["pitch1"].append(ff.pitch1)
        self.buf["pitch2"].append(ff.pitch2)
        self.buf["elbow_flex"].append(ff.elbow_flex)
        self.buf["elbow_vel"].append(ff.elbow_vel)
        self.buf["g1_mag"].append(ff.g1_mag)
        self.buf["g2_mag"].append(ff.g2_mag)
        self.buf["upper_arm_motion"].append(ff.upper_arm_motion)
        self.buf["jerk1"].append(ff.jerk1)
        self.buf["jerk2"].append(ff.jerk2)
        self.buf["a1_mag"].append(ff.a1_mag)
        self.buf["a2_mag"].append(ff.a2_mag)

    def render(self):
        if len(self.t) < 2:
            return
        t_list = list(self.t)
        for name, ybuf in self.buf.items():
            y_list = list(ybuf)
            if len(y_list) != len(t_list):
                continue
            ax = self.ax[name]
            line = self.line[name]
            line.set_data(t_list, y_list)
            ax.relim()
            ax.autoscale_view()
            self.fig[name].canvas.draw_idle()
            self.fig[name].canvas.flush_events()


# -----------------------------
# 3) CSV -> paired IMU frames
# -----------------------------
def load_and_pair_csv(
    csv_path: str,
    imu1_id=1,
    imu2_id=2,
    time_col="t_sec",
    still_calib_seconds=2.0,
    pairing_tolerance_s=0.01,
):

    df = pd.read_csv(csv_path)

    required = [
        time_col, "sensor_id",
        "acc_x_mps2", "acc_y_mps2", "acc_z_mps2",
        "gyro_x_dps", "gyro_y_dps", "gyro_z_dps",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    df = df.sort_values(time_col).reset_index(drop=True)

    df1 = df[df["sensor_id"] == imu1_id].copy().sort_values(time_col)
    df2 = df[df["sensor_id"] == imu2_id].copy().sort_values(time_col)

    if len(df1) == 0 or len(df2) == 0:
        raise ValueError(
            f"Could not find both sensors. Counts: imu1_id={imu1_id} -> {len(df1)}, imu2_id={imu2_id} -> {len(df2)}"
        )

    # --- Still calibration (simple gyro bias removal) using first N seconds
    t0 = min(df1[time_col].iloc[0], df2[time_col].iloc[0])
    calib_end = t0 + float(still_calib_seconds)

    def gyro_bias(dfx):
        win = dfx[(dfx[time_col] >= t0) & (dfx[time_col] <= calib_end)]
        if len(win) < 5:
            return np.array([0.0, 0.0, 0.0], dtype=float)
        return np.array([
            win["gyro_x_dps"].mean(),
            win["gyro_y_dps"].mean(),
            win["gyro_z_dps"].mean(),
        ], dtype=float)

    b1 = gyro_bias(df1)
    b2 = gyro_bias(df2)

    for dfx, b in [(df1, b1), (df2, b2)]:
        dfx["gyro_x_dps"] = dfx["gyro_x_dps"] - b[0]
        dfx["gyro_y_dps"] = dfx["gyro_y_dps"] - b[1]
        dfx["gyro_z_dps"] = dfx["gyro_z_dps"] - b[2]

    # --- Pairing by nearest timestamp (merge_asof)
    # This assumes both streams have roughly the same rate and time base.
    df1p = df1[[time_col,
               "acc_x_mps2","acc_y_mps2","acc_z_mps2",
               "gyro_x_dps","gyro_y_dps","gyro_z_dps"]].copy()
    df2p = df2[[time_col,
               "acc_x_mps2","acc_y_mps2","acc_z_mps2",
               "gyro_x_dps","gyro_y_dps","gyro_z_dps"]].copy()

    df1p = df1p.rename(columns={
        "acc_x_mps2":"ax1","acc_y_mps2":"ay1","acc_z_mps2":"az1",
        "gyro_x_dps":"gx1","gyro_y_dps":"gy1","gyro_z_dps":"gz1",
    })
    df2p = df2p.rename(columns={
        "acc_x_mps2":"ax2","acc_y_mps2":"ay2","acc_z_mps2":"az2",
        "gyro_x_dps":"gx2","gyro_y_dps":"gy2","gyro_z_dps":"gz2",
    })

    paired = pd.merge_asof(
        df1p.sort_values(time_col),
        df2p.sort_values(time_col),
        on=time_col,
        direction="nearest",
        tolerance=pairing_tolerance_s,
    ).dropna()

    # Build frames
    frames = []
    for row in paired.itertuples(index=False):
        frames.append(IMUFrame2(
            t_s=float(getattr(row, time_col)),
            ax1=row.ax1, ay1=row.ay1, az1=row.az1,
            gx1=row.gx1, gy1=row.gy1, gz1=row.gz1,
            ax2=row.ax2, ay2=row.ay2, az2=row.az2,
            gx2=row.gx2, gy2=row.gy2, gz2=row.gz2,
        ))

    return frames


# -----------------------------
# 4) Main loop: play CSV like a "live" stream
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Path to IMU CSV")
    ap.add_argument("--imu1", type=int, default=1, help="sensor_id for IMU1")
    ap.add_argument("--imu2", type=int, default=2, help="sensor_id for IMU2")
    ap.add_argument("--calib_s", type=float, default=2.0, help="seconds at start used for still gyro bias")
    ap.add_argument("--tol", type=float, default=0.01, help="pairing tolerance seconds (merge_asof)")
    ap.add_argument("--seconds", type=float, default=10.0, help="dashboard window seconds")
    ap.add_argument("--sample_hz", type=float, default=100.0, help="approx Hz for dashboard buffer sizing")
    ap.add_argument("--realtime", action="store_true", help="sleep to mimic realtime playback")
    args = ap.parse_args()

    frames = load_and_pair_csv(
        args.csv,
        imu1_id=args.imu1,
        imu2_id=args.imu2,
        still_calib_seconds=args.calib_s,
        pairing_tolerance_s=args.tol,
    )
    if not frames:
        raise RuntimeError("No paired frames produced. Increase --tol or check sensor_id/time base.")

    feat = FeatureEngine(
        alpha=0.98,
        gyro_units="dps",
        pitch_axis_imu1=1,
        pitch_axis_imu2=1,
    )

    dash = LiveDashboard(seconds=args.seconds, sample_hz=args.sample_hz)

    # Normalize to start at 0 for plotting
    t_start = frames[0].t_s
    last_print = 0.0
    prev_t = None

    for fr in frames:
        t_s = fr.t_s - t_start

        ff = feat.update(fr)
        if ff is None:
            continue

        dash.add(t_s, ff)
        dash.render()

        if t_s - last_print > 0.5:
            last_print = t_s
            print(f"elbow_flex={ff.elbow_flex:7.2f} deg | elbow_vel={ff.elbow_vel:7.2f} deg/s")

        # Let matplotlib update
        plt.pause(0.001)

        # Optional realtime playback
        if args.realtime:
            if prev_t is not None:
                dt = (fr.t_s - prev_t)
                if dt > 0:
                    time.sleep(min(dt, 0.05))
            prev_t = fr.t_s

    print("Done.")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
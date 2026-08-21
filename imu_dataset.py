# imu_dataset.py
"""
Turn labeled recording sessions into a windowed feature table for exercise
classification.

This sits on top of imu_batch.process_batch (bias/filter/complementary-filter
per-sample features) and adds the two things a classifier needs that nothing
else in the repo produces yet:

1. Windowing: slice a session's per-sample feature stream into fixed-size,
   overlapping time windows (classification operates on windows of motion,
   not single samples).
2. Labels: attach an exercise label to every window, driven by a manifest
   file (manifest.csv) that maps each recording to the exercise performed,
   rather than any label being inferred from the data itself.

Two raw recording formats are supported (see SESSION_LOADERS):
- "raw_dual": mpu_dual_log.csv / read_mpu_dual.py's native output --
  t_ms,ax1,ay1,az1,gx1,gy1,gz1,ax2,ay2,az2,gx2,gy2,gz2.
- "curl_csv": the example_imu_data_bundle long-format csv -- one row per
  (timestamp, sensor) with acc_*_mps2 / gyro_*_dps columns and a sensor_id
  of "forearm" or "bicep".

Usage:
    python3 imu_dataset.py --manifest manifest.csv --out data/exercise_windows.csv
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import pandas as pd

import imu_batch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Which sensor is which for each format, matching imu_features.py's
# convention: sensor 1 = forearm (pitch1), sensor 2 = upper-arm (pitch2).
FORMAT_GYRO_UNITS = {
    "raw_dual": "raw",
    "curl_csv": "dps",
}

_AGG_FIELDS = [f for f in imu_batch._FIELDS if f not in ("t_ms", "dt")]


# ---------- session loaders: raw file -> (t_ms, a1, g1, a2, g2) ----------

def load_raw_dual_csv(path: str):
    """mpu_dual_log.csv / read_mpu_dual.py format: one row per timestamp,
    both sensors already side by side."""
    rows = []
    with open(path) as f:
        r = csv.reader(f)
        next(r)  # header
        for parts in r:
            rows.append(list(map(float, parts)))
    if not rows:
        raise ValueError(f"{path}: no data rows")
    t_ms = np.array([int(v[0]) for v in rows], dtype=np.int64)
    a1 = np.array([v[1:4] for v in rows])
    g1 = np.array([v[4:7] for v in rows])
    a2 = np.array([v[7:10] for v in rows])
    g2 = np.array([v[10:13] for v in rows])
    return t_ms, a1, g1, a2, g2


def load_curl_csv(path: str):
    """example_imu_data_bundle long-format csv: one row per (t_sec, sensor_id),
    sensor_id in {"forearm", "bicep"}. Pivoted into the wide dual-sensor
    layout process_batch expects (forearm -> sensor 1, bicep -> sensor 2)."""
    df = pd.read_csv(path)
    missing = {"forearm", "bicep"} - set(df["sensor_id"].unique())
    if missing:
        raise ValueError(f"{path}: missing sensor_id(s) {missing}")

    fore = df[df["sensor_id"] == "forearm"].sort_values("t_sec").reset_index(drop=True)
    bicep = df[df["sensor_id"] == "bicep"].sort_values("t_sec").reset_index(drop=True)
    n = min(len(fore), len(bicep))
    fore, bicep = fore.iloc[:n], bicep.iloc[:n]

    t_ms = np.round(fore["t_sec"].to_numpy() * 1000.0).astype(np.int64)
    a1 = fore[["acc_x_mps2", "acc_y_mps2", "acc_z_mps2"]].to_numpy()
    g1 = fore[["gyro_x_dps", "gyro_y_dps", "gyro_z_dps"]].to_numpy()
    a2 = bicep[["acc_x_mps2", "acc_y_mps2", "acc_z_mps2"]].to_numpy()
    g2 = bicep[["gyro_x_dps", "gyro_y_dps", "gyro_z_dps"]].to_numpy()
    return t_ms, a1, g1, a2, g2


SESSION_LOADERS = {
    "raw_dual": load_raw_dual_csv,
    "curl_csv": load_curl_csv,
}


# ---------- windowing ----------

def iter_windows(t_ms, window_ms: float, hop_ms: float):
    """Yield (lo, hi) half-open index ranges into t_ms, one per time window.
    Windows are placed by time (not sample count) so this is correct even if
    the sensor's sample rate drifts or has gaps."""
    t_ms = np.asarray(t_ms)
    if len(t_ms) == 0:
        return
    start = float(t_ms[0])
    t_end = float(t_ms[-1])
    while start + window_ms <= t_end + 1e-6:
        stop = start + window_ms
        idx = np.where((t_ms >= start) & (t_ms < stop))[0]
        if len(idx) > 0:
            yield int(idx[0]), int(idx[-1]) + 1
        start += hop_ms


def aggregate_window(feat: dict, lo: int, hi: int) -> dict:
    """Collapse one window of per-sample features into a fixed-size row of
    summary statistics -- mean/std/min/max/rms per feature channel."""
    row = {
        "n_samples": hi - lo,
        "duration_s": (feat["t_ms"][hi - 1] - feat["t_ms"][lo]) / 1000.0,
    }
    for name in _AGG_FIELDS:
        vals = feat[name][lo:hi]
        row[f"{name}_mean"] = float(np.mean(vals))
        row[f"{name}_std"] = float(np.std(vals))
        row[f"{name}_min"] = float(np.min(vals))
        row[f"{name}_max"] = float(np.max(vals))
        row[f"{name}_rms"] = float(np.sqrt(np.mean(vals ** 2)))
    return row


# ---------- manifest-driven dataset build ----------

def load_manifest(manifest_path: str) -> pd.DataFrame:
    df = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    required = {"session_id", "file", "format", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{manifest_path}: missing column(s) {missing}")
    return df


def build_dataset(
    manifest_path: str,
    repo_root: str = REPO_ROOT,
    window_sec: float = 2.0,
    hop_sec: float = 1.0,
    min_samples: int = 5,
) -> pd.DataFrame:
    manifest = load_manifest(manifest_path)
    window_ms, hop_ms = window_sec * 1000.0, hop_sec * 1000.0

    all_rows = []
    for _, session in manifest.iterrows():
        if not session["label"].strip():
            print(f"skip {session['session_id']!r}: no label in manifest")
            continue
        fmt = session["format"].strip()
        loader = SESSION_LOADERS.get(fmt)
        if loader is None:
            raise ValueError(
                f"{session['session_id']!r}: unknown format {fmt!r}, "
                f"expected one of {sorted(SESSION_LOADERS)}"
            )

        path = os.path.join(repo_root, session["file"])
        t_ms, a1, g1, a2, g2 = loader(path)
        feat = imu_batch.process_batch(
            t_ms, a1, g1, a2, g2, gyro_units=FORMAT_GYRO_UNITS[fmt]
        )
        if len(feat["t_ms"]) == 0:
            print(f"skip {session['session_id']!r}: no usable samples after feature extraction")
            continue

        n_windows = 0
        for w, (lo, hi) in enumerate(iter_windows(feat["t_ms"], window_ms, hop_ms)):
            if hi - lo < min_samples:
                continue
            row = aggregate_window(feat, lo, hi)
            row.update({
                "session_id": session["session_id"],
                "window_index": w,
                "label": session["label"],
                "subject": session.get("subject", ""),
            })
            all_rows.append(row)
            n_windows += 1
        print(f"{session['session_id']!r} ({session['label']}): {n_windows} windows")

    if not all_rows:
        raise ValueError("no windows produced -- check manifest labels and window_sec/hop_sec")

    df = pd.DataFrame(all_rows)
    meta_cols = ["session_id", "window_index", "label", "subject", "n_samples", "duration_s"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    return df[meta_cols + feature_cols]


def build_sequence_dataset(
    manifest_path: str,
    repo_root: str = REPO_ROOT,
    window_sec: float = 2.0,
    hop_sec: float = 1.0,
    min_samples: int = 5,
):
    """Like build_dataset, but returns raw per-sample feature sequences instead
    of aggregated summary statistics -- the input a sequence model (e.g. an
    LSTM) needs, as opposed to the per-window statistics classical models use.

    Windows vary by a sample or two in length near session boundaries, so
    every window is truncated to the shortest length seen, giving one fixed
    sequence length across the whole dataset.

    Returns (X, y, meta): X is (n_windows, seq_len, n_features) float32 (the
    same _AGG_FIELDS feature order as build_dataset's column names, minus the
    per-window aggregation), y is (n_windows,) label strings, and meta is a
    DataFrame with session_id / window_index / label / subject, aligned
    row-for-row with X and y.
    """
    manifest = load_manifest(manifest_path)
    window_ms, hop_ms = window_sec * 1000.0, hop_sec * 1000.0

    windows = []  # (lo, hi, feat, session_id, window_index, label, subject)
    for _, session in manifest.iterrows():
        if not session["label"].strip():
            continue
        fmt = session["format"].strip()
        loader = SESSION_LOADERS.get(fmt)
        if loader is None:
            raise ValueError(
                f"{session['session_id']!r}: unknown format {fmt!r}, "
                f"expected one of {sorted(SESSION_LOADERS)}"
            )

        path = os.path.join(repo_root, session["file"])
        t_ms, a1, g1, a2, g2 = loader(path)
        feat = imu_batch.process_batch(
            t_ms, a1, g1, a2, g2, gyro_units=FORMAT_GYRO_UNITS[fmt]
        )
        if len(feat["t_ms"]) == 0:
            continue

        for w, (lo, hi) in enumerate(iter_windows(feat["t_ms"], window_ms, hop_ms)):
            if hi - lo < min_samples:
                continue
            windows.append(
                (lo, hi, feat, session["session_id"], w, session["label"], session.get("subject", ""))
            )

    if not windows:
        raise ValueError("no windows produced -- check manifest labels and window_sec/hop_sec")

    seq_len = min(hi - lo for lo, hi, *_ in windows)
    X = np.stack([
        np.stack([feat[name][lo:lo + seq_len] for name in _AGG_FIELDS], axis=-1)
        for lo, hi, feat, *_ in windows
    ]).astype(np.float32)
    y = np.array([w[5] for w in windows])
    meta = pd.DataFrame({
        "session_id": [w[3] for w in windows],
        "window_index": [w[4] for w in windows],
        "label": y,
        "subject": [w[6] for w in windows],
    })
    return X, y, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=os.path.join(REPO_ROOT, "manifest.csv"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "data", "exercise_windows.csv"))
    ap.add_argument("--window-sec", type=float, default=2.0)
    ap.add_argument("--hop-sec", type=float, default=1.0)
    args = ap.parse_args()

    df = build_dataset(args.manifest, window_sec=args.window_sec, hop_sec=args.hop_sec)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} windows x {len(df.columns)} columns -> {args.out}")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()

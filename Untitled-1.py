"""
Pose classifier (camera) using MediaPipe Pose + simple rule-based heuristics.

What it does:
- Opens your webcam
- Runs MediaPipe Pose
- Computes joint angles (elbows, knees, hips, shoulders)
- Classifies a few common poses/exercises:
  - Standing
  - Squat (down / up-ish)
  - Pushup / Plank
  - Bicep Curl (left / right)
  - Overhead Press (arms overhead)
  Biceps curl (elbow flexion/extension)

Triceps extension / kickback (elbow extension dominant; different timing + direction from curl)

Hammer curl vs regular curl (often doable if sensor orientation is stable; otherwise moderate)

Shoulder press (single-arm dumbbell) (shoulder flexion/abduction + elbow extension; distinct multi-joint pattern)

Lateral raise (shoulder abduction; elbow mostly fixed)

Front raise (shoulder flexion; different plane vs lateral raise)
 necessary 

Notes:
- This is NOT a trained model; it’s a strong baseline for a portfolio demo.
- Works best when your full body is visible (or at least upper body for pushups/curls).
"""

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple


import cv2

try:
    import mediapipe as mp
except ImportError:
    raise SystemExit(
        "mediapipe not found. Install with:\n"
        "  pip install mediapipe opencv-python\n"
        "If you are on Apple Silicon and hit issues, try:\n"
        "  pip install mediapipe==0.10.14 opencv-python\n"
    )

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# -------------------- Geometry helpers --------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def angle_abc(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    """
    Returns angle at point b (in degrees) formed by points a-b-c.
    """
    bax = a[0] - b[0]
    bay = a[1] - b[1]
    bcx = c[0] - b[0]
    bcy = c[1] - b[1]

    dot = bax * bcx + bay * bcy
    norm1 = math.hypot(bax, bay)
    norm2 = math.hypot(bcx, bcy)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return float("nan")
    cosv = clamp(dot / (norm1 * norm2), -1.0, 1.0)
    return math.degrees(math.acos(cosv))

def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def is_valid(v: float) -> bool:
    return not (math.isnan(v) or math.isinf(v))

def xy(lm) -> Tuple[float, float]:
    return (lm.x, lm.y)


# -------------------- Pose extraction --------------------

@dataclass
class Keypoints:
    # only the points we need (normalized coords [0..1])
    l_sh: Tuple[float, float]
    r_sh: Tuple[float, float]
    l_el: Tuple[float, float]
    r_el: Tuple[float, float]
    l_wr: Tuple[float, float]
    r_wr: Tuple[float, float]
    l_hip: Tuple[float, float]
    r_hip: Tuple[float, float]
    l_kn: Tuple[float, float]
    r_kn: Tuple[float, float]
    l_an: Tuple[float, float]
    r_an: Tuple[float, float]

    # visibility (for quick confidence checks)
    v_min: float

def extract_keypoints(landmarks) -> Optional[Keypoints]:
    """
    landmarks: result.pose_landmarks.landmark
    """
    # indices from MediaPipe Pose Landmarks
    # https://developers.google.com/mediapipe/solutions/vision/pose_landmarker

    L_SH, R_SH = 11, 12
    L_EL, R_EL = 13, 14
    L_WR, R_WR = 15, 16
    L_HIP, R_HIP = 23, 24
    L_KN, R_KN = 25, 26
    L_AN, R_AN = 27, 28

    idxs = [L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN]
    vis = [landmarks[i].visibility for i in idxs]
    v_min = min(vis)

    # If pose is too uncertain, bail (you can lower this threshold)
    if v_min < 0.35:
        return None

    return Keypoints(
        l_sh=xy(landmarks[L_SH]),
        r_sh=xy(landmarks[R_SH]),
        l_el=xy(landmarks[L_EL]),
        r_el=xy(landmarks[R_EL]),
        l_wr=xy(landmarks[L_WR]),
        r_wr=xy(landmarks[R_WR]),
        l_hip=xy(landmarks[L_HIP]),
        r_hip=xy(landmarks[R_HIP]),
        l_kn=xy(landmarks[L_KN]),
        r_kn=xy(landmarks[R_KN]),
        l_an=xy(landmarks[L_AN]),
        r_an=xy(landmarks[R_AN]),
        v_min=v_min,
    )


# -------------------- Simple classifier --------------------

@dataclass
class PoseFeatures:
    l_elbow: float
    r_elbow: float
    l_knee: float
    r_knee: float
    l_hip: float
    r_hip: float
    torso_angle: float  # angle at hip between shoulder-hip-knee (per side averaged)
    shoulder_width: float
    hip_width: float

def compute_features(kp: Keypoints) -> PoseFeatures:
    # elbows
    l_elbow = angle_abc(kp.l_sh, kp.l_el, kp.l_wr)
    r_elbow = angle_abc(kp.r_sh, kp.r_el, kp.r_wr)

    # knees

    l_knee = angle_abc(kp.l_hip, kp.l_kn, kp.l_an)
    r_knee = angle_abc(kp.r_hip, kp.r_kn, kp.r_an)

    # hips (shoulder-hip-knee)
    l_hip = angle_abc(kp.l_sh, kp.l_hip, kp.l_kn)
    r_hip = angle_abc(kp.r_sh, kp.r_hip, kp.r_kn)

    torso_angle = (l_hip + r_hip) / 2.0 if (is_valid(l_hip) and is_valid(r_hip)) else (l_hip if is_valid(l_hip) else r_hip)

    shoulder_width = dist(kp.l_sh, kp.r_sh)
    hip_width = dist(kp.l_hip, kp.r_hip)

    return PoseFeatures(
        l_elbow=l_elbow, r_elbow=r_elbow,
        l_knee=l_knee, r_knee=r_knee,
        l_hip=l_hip, r_hip=r_hip,
        torso_angle=torso_angle,
        shoulder_width=shoulder_width,
        hip_width=hip_width
    )

def classify_pose(kp: Keypoints, f: PoseFeatures) -> str:
    """
    Heuristic classification. Tune thresholds as needed.
    """
    # Helper stats
    elbow_min = min(f.l_elbow, f.r_elbow)
    elbow_max = max(f.l_elbow, f.r_elbow)
    knee_min = min(f.l_knee, f.r_knee)
    knee_max = max(f.l_knee, f.r_knee)

    # "Arms overhead" check: wrists above shoulders (y smaller = higher in image)
    wrists_overhead = (kp.l_wr[1] < kp.l_sh[1] - 0.05) and (kp.r_wr[1] < kp.r_sh[1] - 0.05)

    # Basic “body horizontal” check: shoulders and hips roughly same y-level
    shoulders_level = abs(kp.l_sh[1] - kp.r_sh[1]) < 0.08
    hips_level = abs(kp.l_hip[1] - kp.r_hip[1]) < 0.10
    # if shoulders and hips are at similar vertical level, person may be horizontal (pushup/plank)
    body_horizontalish = shoulders_level and hips_level and abs(((kp.l_sh[1] + kp.r_sh[1]) / 2) - ((kp.l_hip[1] + kp.r_hip[1]) / 2)) < 0.08

    # Squat: knees bent + hips bent while mostly upright (not horizontal)
    is_squat_down = (knee_min < 120) and (min(f.l_hip, f.r_hip) < 120) and (not body_horizontalish)
    is_squat_upish = (knee_min > 150) and (min(f.l_hip, f.r_hip) > 150) and (not body_horizontalish)

    # Pushup/plank: body horizontal-ish + elbows somewhat straight (plank) or bent (pushup)
    is_plank = body_horizontalish and (elbow_min > 150)
    is_pushup = body_horizontalish and (elbow_min < 120)

    # Curl: one elbow significantly bent while that wrist is below shoulder and near torso

    # (This is crude; good enough to demo.)
    left_curl = (f.l_elbow < 70) and (kp.l_wr[1] > kp.l_sh[1]) and (dist(kp.l_wr, kp.l_el) < 0.20)
    right_curl = (f.r_elbow < 70) and (kp.r_wr[1] > kp.r_sh[1]) and (dist(kp.r_wr, kp.r_el) < 0.20)

    # Overhead press / hands up
    if wrists_overhead and elbow_min > 140:
        return "Overhead Press / Arms Overhead"

    if is_pushup:
        return "Pushup (Down)"
    if is_plank:
        return "Plank / Pushup (Up)"

    if is_squat_down:
        return "Squat (Down)"
    if is_squat_upish:
        return "Standing / Squat (Up)"

    if left_curl and right_curl:
        return "Bicep Curl (Both)"
    if left_curl:
        return "Bicep Curl (Left)"
    if right_curl:
        return "Bicep Curl (Right)"

    # Fallbacks
    if knee_min > 150 and elbow_min > 150:
        return "Standing (Arms Straight)"
    return "Unknown"


# -------------------- Visualization --------------------

def draw_label(img, text: str, x: int = 20, y: int = 40):

    cv2.rectangle(img, (x - 10, y - 30), (x + 520, y + 10), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

def draw_debug(img, f: PoseFeatures, x: int = 20, y: int = 80):
    lines = [
        f"Elbows L/R: {f.l_elbow:5.1f} / {f.r_elbow:5.1f}",
        f"Knees  L/R: {f.l_knee:5.1f} / {f.r_knee:5.1f}",
        f"Hips   L/R: {f.l_hip:5.1f} / {f.r_hip:5.1f}",
    ]
    yy = y
    for ln in lines:
        cv2.putText(img, ln, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        yy += 24


# -------------------- Main loop --------------------

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam (index 0). Try changing VideoCapture(0) to (1).")

    # You can increase model_complexity for accuracy (0/1/2) but it may slow down
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        last = time.time()
        fps = 0.0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            label = "No pose detected"
            if res.pose_landmarks:
                kp = extract_keypoints(res.pose_landmarks.landmark)
                if kp is not None:
                    feats = compute_features(kp)
                    label = classify_pose(kp, feats)
                    mp_drawing.draw_landmarks(
                        frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS
                    )
                    draw_debug(frame, feats)
            

            now = time.time()
            dt = now - last
            last = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            draw_label(frame, f"{label}   |   FPS: {fps:.1f}")

            cv2.imshow("MediaPipe Pose Classifier", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):  # ESC or q
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

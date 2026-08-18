import threading, time
import serial
import math
import pygame
from imu_serial import IMUStream, accel_to_roll_pitch


# ---------------- Math helpers ----------------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def vec_norm(v):
    return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])

def vec_unit(v):
    n = vec_norm(v)
    if n < 1e-9:
        return (0.0, 1.0, 0.0)
    return (v[0]/n, v[1]/n, v[2]/n)

def rot_x(a):
    ca, sa = math.cos(a), math.sin(a)
    return ((1, 0, 0), (0, ca, -sa), (0, sa, ca))

def rot_y(a):
    ca, sa = math.cos(a), math.sin(a)
    return ((ca, 0, sa), (0, 1, 0), (-sa, 0, ca))

def rot_z(a):
    ca, sa = math.cos(a), math.sin(a)
    return ((ca, -sa, 0), (sa, ca, 0), (0, 0, 1))

defs = {}

def mat_mul(A, B):
    return tuple(
        tuple(sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )

def mat_T(M):
    return (
        (M[0][0], M[1][0], M[2][0]),
        (M[0][1], M[1][1], M[2][1]),
        (M[0][2], M[1][2], M[2][2]),
    )

def mat_vec(M, v):
    return (
        M[0][0]*v[0] + M[0][1]*v[1] + M[0][2]*v[2],
        M[1][0]*v[0] + M[1][1]*v[1] + M[1][2]*v[2],
        M[2][0]*v[0] + M[2][1]*v[1] + M[2][2]*v[2],
    )


# ---------------- Complementary filter IMU ----------------
class ComplementaryIMU:
    """
    roll/pitch stabilized by accel, yaw gyro integrated (drifts).
    """
    def __init__(self, alpha=0.98, gyro_lsb_per_dps=131.0):
        self.alpha = alpha
        self.gyro_lsb_per_dps = gyro_lsb_per_dps

        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        self.bgx = 0.0
        self.bgy = 0.0
        self.bgz = 0.0

        # reference rotation at zero pose
        self.R0 = ((1.0, 0.0, 0.0),
                   (0.0, 1.0, 0.0),
                   (0.0, 0.0, 1.0))
        self.initialized = False

    def update(self, ax, ay, az, gx, gy, gz, dt):
        a_roll, a_pitch = accel_to_roll_pitch(ax, ay, az)

        gx = gx - self.bgx
        gy = gy - self.bgy
        gz = gz - self.bgz

        gx_rs = (gx / self.gyro_lsb_per_dps) * (math.pi / 180.0)
        gy_rs = (gy / self.gyro_lsb_per_dps) * (math.pi / 180.0)
        gz_rs = (gz / self.gyro_lsb_per_dps) * (math.pi / 180.0)

        if not self.initialized:
            self.roll, self.pitch, self.yaw = a_roll, a_pitch, 0.0
            self.initialized = True
            return

        self.roll += gx_rs * dt
        self.pitch += gy_rs * dt
        self.yaw += gz_rs * dt

        self.roll = self.alpha * self.roll + (1.0 - self.alpha) * a_roll
        self.pitch = self.alpha * self.pitch + (1.0 - self.alpha) * a_pitch

        if self.yaw > math.pi:
            self.yaw -= 2 * math.pi
        elif self.yaw < -math.pi:
            self.yaw += 2 * math.pi

    def R_current(self):
        # R = Rz(yaw) * Ry(pitch) * Rx(roll)
        return mat_mul(rot_z(self.yaw), mat_mul(rot_y(self.pitch), rot_x(self.roll)))

    def R_relative(self):
        # relative to zero pose: R_rel = R0^T * R
        return mat_mul(mat_T(self.R0), self.R_current())

    def calibrate(self, samples):
        """
        samples: list[(ax,ay,az,gx,gy,gz)] while held still in your chosen zero pose
        """
        if len(samples) < 10:
            return False

        self.bgx = sum(s[3] for s in samples) / len(samples)
        self.bgy = sum(s[4] for s in samples) / len(samples)
        self.bgz = sum(s[5] for s in samples) / len(samples)

        # baseline tilt from accel average
        axm = sum(s[0] for s in samples) / len(samples)
        aym = sum(s[1] for s in samples) / len(samples)
        azm = sum(s[2] for s in samples) / len(samples)
        r0, p0 = accel_to_roll_pitch(axm, aym, azm)

        self.roll, self.pitch, self.yaw = r0, p0, 0.0
        self.R0 = self.R_current()
        self.initialized = True
        return True


# ---------------- Drawing helpers ----------------
def project_to_screen(v3, origin):
    # orthographic: x right, y up (we invert for screen)
    x, y, z = v3
    return (int(origin[0] + x), int(origin[1] - y))

def draw_thick_line(surf, color, p1, p2, width):
    pygame.draw.line(surf, color, p1, p2, width)

def draw_person_static(screen, mid, scale=1.0):
    """
    Draw a simple stick person for reference.
    Returns key joint anchor points (in screen coords).
    """
    mx, my = mid

    head_r = int(28 * scale)
    neck_to_hip = int(140 * scale)
    shoulder_half = int(65 * scale)
    hip_half = int(40 * scale)
    leg_len = int(150 * scale)
    arm_len = int(110 * scale)

    head_center = (mx, my - int(200 * scale))
    neck = (mx, my - int(160 * scale))
    shoulder_center = (mx, my - int(140 * scale))
    hip_center = (mx, my - int(140 * scale) + neck_to_hip)

    left_shoulder = (shoulder_center[0] - shoulder_half, shoulder_center[1])
    right_shoulder = (shoulder_center[0] + shoulder_half, shoulder_center[1])

    left_hip = (hip_center[0] - hip_half, hip_center[1])
    right_hip = (hip_center[0] + hip_half, hip_center[1])

    left_knee = (left_hip[0] - int(10 * scale), left_hip[1] + int(leg_len * 0.55))
    right_knee = (right_hip[0] + int(10 * scale), right_hip[1] + int(leg_len * 0.55))
    left_ankle = (left_knee[0] - int(5 * scale), left_knee[1] + int(leg_len * 0.50))
    right_ankle = (right_knee[0] + int(5 * scale), right_knee[1] + int(leg_len * 0.50))

    # torso + head
    pygame.draw.circle(screen, (235, 235, 245), head_center, head_r, 3)
    draw_thick_line(screen, (200, 200, 220), neck, hip_center, 8)
    draw_thick_line(screen, (200, 200, 220), left_shoulder, right_shoulder, 8)
    draw_thick_line(screen, (200, 200, 220), left_hip, right_hip, 8)

    # left arm static (just a default down-ish)
    left_elbow = (left_shoulder[0] - int(20 * scale), left_shoulder[1] + int(arm_len * 0.55))
    left_wrist = (left_elbow[0] - int(10 * scale), left_elbow[1] + int(arm_len * 0.55))
    draw_thick_line(screen, (170, 170, 190), left_shoulder, left_elbow, 8)
    draw_thick_line(screen, (170, 170, 190), left_elbow, left_wrist, 8)
    pygame.draw.circle(screen, (235, 235, 245), left_elbow, int(6*scale))
    pygame.draw.circle(screen, (235, 235, 245), left_wrist, int(5*scale))

    # legs static
    draw_thick_line(screen, (170, 170, 190), left_hip, left_knee, 10)
    draw_thick_line(screen, (170, 170, 190), left_knee, left_ankle, 10)
    draw_thick_line(screen, (170, 170, 190), right_hip, right_knee, 10)
    draw_thick_line(screen, (170, 170, 190), right_knee, right_ankle, 10)

    # joints
    pygame.draw.circle(screen, (235, 235, 245), left_shoulder, int(6*scale))
    pygame.draw.circle(screen, (235, 235, 245), right_shoulder, int(6*scale))
    pygame.draw.circle(screen, (235, 235, 245), left_hip, int(6*scale))
    pygame.draw.circle(screen, (235, 235, 245), right_hip, int(6*scale))

    return {
        "right_shoulder": right_shoulder,
        "shoulder_center": shoulder_center,
        "hip_center": hip_center,
    }


def main():
    # ---- set your serial port here ----
    PORT = "/dev/cu.usbserial-0001"  
    BAUD = 115200

    stream = IMUStream(PORT, BAUD)

    pygame.init()
    W, H = 1100, 750
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Person model: right arm driven by 2x MPU6050 (C = calibrate)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 22)

    # Two IMUs: 1=upper arm, 2=forearm
    imu_u = ComplementaryIMU(alpha=0.98)
    imu_f = ComplementaryIMU(alpha=0.98)

    # Choose bone axis in SENSOR local frame.
    # If it looks wrong, try (1,0,0), (0,0,1), or flip sign.
    bone_axis_local = (0.0, 1.0, 0.0)

    # lengths (pixels)
    Lu = 150
    Lf = 135

    last_t_ms = None

    def dt_from_vals(vals):
        nonlocal last_t_ms
        t_ms = vals[0]
        if last_t_ms is None:
            last_t_ms = t_ms
            return 1/100.0
        dt = (t_ms - last_t_ms) / 1000.0
        last_t_ms = t_ms
        return clamp(dt, 1e-4, 0.05)

    def calibrate(seconds=1.0):
        # gather samples while held still
        t0 = time.time()
        su, sf = [], []
        while time.time() - t0 < seconds:
            vals = stream.get_latest()
            if not vals:
                continue
            _, ax1, ay1, az1, gx1, gy1, gz1, ax2, ay2, az2, gx2, gy2, gz2 = vals
            su.append((ax1, ay1, az1, gx1, gy1, gz1))
            sf.append((ax2, ay2, az2, gx2, gy2, gz2))
            time.sleep(0.005)
        ok1 = imu_u.calibrate(su)
        ok2 = imu_f.calibrate(sf)
        return ok1 and ok2

    # cached directions
    du = (0.0, -1.0, 0.0)
    df = (0.0, -1.0, 0.0)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_c:
                    screen.fill((15, 16, 20))
                    msg = font.render("Calibrating... hold your RIGHT arm still for ~1 second", True, (235, 235, 245))
                    screen.blit(msg, (12, 12))
                    pygame.display.flip()
                    ok = calibrate(1.0)
                    print("Calibration:", "OK" if ok else "FAILED")

        vals = stream.get_latest()
        if vals:
            dt = dt_from_vals(vals)
            _, ax1, ay1, az1, gx1, gy1, gz1, ax2, ay2, az2, gx2, gy2, gz2 = vals

            imu_u.update(ax1, ay1, az1, gx1, gy1, gz1, dt)
            imu_f.update(ax2, ay2, az2, gx2, gy2, gz2, dt)

            Ru = imu_u.R_relative()
            Rf = imu_f.R_relative()

            du = vec_unit(mat_vec(Ru, bone_axis_local))
            df = vec_unit(mat_vec(Rf, bone_axis_local))

        # ---- draw scene ----
        screen.fill((15, 16, 20))
        joints = draw_person_static(screen, mid=(W//2, H//2+120), scale=1.0)

        right_shoulder = joints["right_shoulder"]

        # Build arm points from 3D directions, projected into screen:
        elbow_3d = (du[0]*Lu, du[1]*Lu, du[2]*Lu)
        elbow_2d = project_to_screen(elbow_3d, right_shoulder)

        wrist_3d = (elbow_3d[0] + df[0]*Lf,
                    elbow_3d[1] + df[1]*Lf,
                    elbow_3d[2] + df[2]*Lf)
        wrist_2d = project_to_screen(wrist_3d, right_shoulder)

        # draw right arm (animated)
        draw_thick_line(screen, (220, 220, 235), right_shoulder, elbow_2d, 12)
        pygame.draw.circle(screen, (245, 245, 255), elbow_2d, 7)
        draw_thick_line(screen, (220, 220, 235), elbow_2d, wrist_2d, 12)
        pygame.draw.circle(screen, (245, 245, 255), wrist_2d, 6)

        # HUD
        hud = font.render("C=calibrate (hold arm still), ESC=quit", True, (180, 180, 190))
        screen.blit(hud, (12, 10))
        hud2 = font.render(
            f"du=({du[0]:+.2f},{du[1]:+.2f},{du[2]:+.2f})  df=({df[0]:+.2f},{df[1]:+.2f},{df[2]:+.2f})",
            True, (160, 160, 175)
        )
        screen.blit(hud2, (12, 34))

        pygame.display.flip()
        clock.tick(60)

    stream.close()
    pygame.quit()


if __name__ == "__main__":
    main()
from object_3d import *
from camera import *
from projection import *
import pygame as pg
import math
from imu_serial import IMUStream, accel_to_roll_pitch



class SoftwareRender:
    def __init__(self):
        pg.init()
        self.RES = self.WIDTH, self.HEIGHT = 1600, 900
        self.H_WIDTH, self.H_HEIGHT = self.WIDTH // 2, self.HEIGHT // 2
        self.FPS = 60
        self.screen = pg.display.set_mode(self.RES)
        self.clock = pg.time.Clock()
        self.create_objects()
        self.imu = IMUStream(port="/dev/cu.usbserial-0001", baud=115200)  # <- your port
        self.last_angles = (0.0, 0.0)  # roll,pitch
        self.yaw = 0.0
        self.last_t = None
        self.gyro_lsb_per_dps = 131.0 # decrease for lower yaw


    def make_box(self):
        w, h, d = 2.0, 1.0, 0.4
        v = [
            [-w/2, -h/2, -d/2, 1],
            [ w/2, -h/2, -d/2, 1],
            [ w/2,  h/2, -d/2, 1],
            [-w/2,  h/2, -d/2, 1],
            [-w/2, -h/2,  d/2, 1],
            [ w/2, -h/2,  d/2, 1],
            [ w/2,  h/2,  d/2, 1],
            [-w/2,  h/2,  d/2, 1],
        ]
        faces = [
            [0,1,2,3],
            [4,5,6,7],
            [0,1,5,4],
            [2,3,7,6],
            [1,2,6,5],
            [0,3,7,4],
        ]
        return Object3D(self, v, faces)

    def create_objects(self):
        self.camera = Camera(self, [0, 0, -10])
        self.projection = Projection(self)
        self.object = self.make_box()

    def get_object_from_file(self, filename):
        vertex, faces = [], []
        with open(filename) as f:
            for line in f:
                if line.startswith('v '):
                    vertex.append([float(i) for i in line.split()[1:]] + [1])
                elif line.startswith('f'):
                    faces_ = line.split()[1:]
                    faces.append([int(face_.split('/')[0]) - 1 for face_ in faces_])
        return Object3D(self, vertex, faces)

    def draw(self):
        self.screen.fill(pg.Color('darkslategray'))
        self.object.draw()

    def run(self):
        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.imu.close()
                    pg.quit()
                    raise SystemExit
            sample = self.imu.get_latest()
            if sample:
                t, ax1, ay1, az1, gx1, gy1, gz1, ax2, ay2, az2, gx2, gy2, gz2 = sample

                # ----- dt in seconds -----
                if self.last_t is None:
                    self.last_t = t
                dt = (t - self.last_t) / 1000.0
                self.last_t = t
                if dt < 0 or dt > 0.2:   # guard against weird jumps/resets
                    dt = 0.//0
                # ----- roll/pitch from accel (stable) -----
                roll, pitch = accel_to_roll_pitch(ax1, ay1, az1)

                alpha = 0.15
                r0, p0 = self.last_angles
                roll  = (1 - alpha) * r0 + alpha * roll
                pitch = (1 - alpha) * p0 + alpha * pitch
                self.last_angles = (roll, pitch)

                # ----- yaw from gyro integration -----
                # convert raw gz -> deg/s -> rad/s
                gz_dps = gz1 / self.gyro_lsb_per_dps
                gz_rads = gz_dps * (math.pi / 180.0)

                # integrate
                self.yaw += gz_rads * dt

                # (optional) keep yaw bounded so it doesn't grow forever
                if self.yaw > math.pi:
                    self.yaw -= 2 * math.pi
                elif self.yaw < -math.pi:
                    self.yaw += 2 * math.pi

                # ----- apply to object: reset then rotate -----
                self.object.vertices = self.object.base_vertices.copy()
                self.object.rotate_z(self.yaw)     # yaw (spin flat)
                self.object.rotate_x(roll)         # tilt
                self.object.rotate_y(pitch)        # tilt

            self.draw()
            self.camera.control()
            pg.display.set_caption(str(self.clock.get_fps()))
            pg.display.flip()
            self.clock.tick(self.FPS)


if __name__ == '__main__':
    app = SoftwareRender()
    app.run()
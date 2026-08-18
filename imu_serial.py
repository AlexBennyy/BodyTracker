import threading, time
import serial
import math

class IMUStream:
    def __init__(self, port, baud=115200):
        self.ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)
        self.lock = threading.Lock()
        self.latest = None
        self._stop = False
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def close(self):
        self._stop = True
        try: self.ser.close()
        except: pass

    def _loop(self):
        while not self._stop:
            line = self.ser.readline().decode("utf-8", errors="ignore").strip()
            if not line or line.startswith("t_ms") or line.startswith("ERROR:"):
                continue
            parts = line.split(",")
            
            # expecting: t_ms,ax1,ay1,az1,gx1,gy1,gz1,ax2,ay2,az2,gx2,gy2,gz2
            if len(parts) != 13:
                continue
            try:
                vals = list(map(int, parts))
            except:
                continue
            with self.lock:
                self.latest = vals

    def get_latest(self):
        with self.lock:
            return self.latest

def accel_to_roll_pitch(ax, ay, az):
    # ax,ay,az are raw int16; scale cancels out for angles
    # roll around X, pitch around Y (one common convention) 
    roll = math.atan2(ay, az)
    pitch = math.atan2(-ax, math.sqrt(ay*ay + az*az))
    return roll, pitch

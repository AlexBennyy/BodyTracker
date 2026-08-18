# run_preprocess_test.py
import time
from imu_preprocess import IMUPreprocessor


imu = IMUPreprocessor("/dev/cu.usbserial-0001", baud=115200, accel_units="raw", gyro_units="raw")

print("Stand still in neutral pose… calibrating")
time.sleep(1.0)
imu.calibrate_still(seconds=3.0)
print("Calibrated.")

while True:
    fr = imu.get_latest()
    if fr:
        print(fr.t_ms, fr.a1, fr.g1, fr.a2, fr.g2)
    time.sleep(0.05)


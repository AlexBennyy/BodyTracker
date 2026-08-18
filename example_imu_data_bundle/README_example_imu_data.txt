Example IMU data (two sensors) — bicep + forearm

This bundle contains:
1) example_imu_two_sensors_curl_50hz.csv
2) example_imu_two_sensors_curl_50hz.jsonl

What it represents:
- 20 seconds of data at 50 Hz (so 1000 samples per sensor, 2000 total rows).
- A smooth "bicep curl" motion: forearm rotates 0→90°→0 repeatedly (~1 rep per 4 seconds).
- Columns include raw-like IMU streams (acc/gyro/mag) plus a synthetic orientation quaternion
  (quat_w, quat_x, quat_y, quat_z) that you might get from an onboard sensor-fusion filter.
- Units:
  - acc_* : meters per second^2 (m/s^2)
  - gyro_*: degrees per second (deg/s)
  - mag_* : microtesla (uT)

Typical BLE payloads:
- Real BLE packets often send fewer fields (e.g., acc+gyro only), in fixed-point integers
  (like int16) scaled by a factor. You can treat this dataset as the "decoded" / engineering-units
  version after applying scale factors.

Quick parsing tips:
- CSV: load with pandas read_csv
- JSONL: read line-by-line; each line is one measurement dict

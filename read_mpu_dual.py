import time
import csv
import serial

PORT = "/dev/cu.usbserial-0001"   # <-- change if needed
BAUD = 115200
OUT_CSV = "mpu_dual_log.csv"

COLUMNS = ["t_ms","ax1","ay1","az1","gx1","gy1","gz1","ax2","ay2","az2","gx2","gy2","gz2"]

def parse_line(line: str):
    line = line.strip()
    if not line:
        return None
    if line.startswith("t_ms,"):
        return None
    if line.startswith("ERROR:"):
        print(line)
        return None

    parts = line.split(",")
    if len(parts) != len(COLUMNS):
        return None

    try:
        vals = list(map(int, parts))
    except ValueError:
        return None

    return dict(zip(COLUMNS, vals))

def inference_step(row: dict):
    # Example "inference": accel magnitude from sensor 1
    ax, ay, az = row["ax1"], row["ay1"], row["az1"]
    return (ax*ax + ay*ay + az*az) ** 0.5

def main():
    print(f"Opening {PORT} @ {BAUD}...")
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)  # ESP32 often resets when serial opens

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()

        print("Reading... Ctrl+C to stop.")
        try:
            while True:
                raw = ser.readline().decode("utf-8", errors="ignore")
                row = parse_line(raw)
                if row is None:
                    continue

                score = inference_step(row)
                print(f"t={row['t_ms']:>6} ax1={row['ax1']:>6} ax2={row['ax2']:>6} score={score:>10.2f}")

                writer.writerow(row)
                f.flush()

        except KeyboardInterrupt:
            print("\nStopping... saved CSV.")
        finally:
            ser.close()
            print(f"Saved: {OUT_CSV}")

if __name__ == "__main__":
    main()

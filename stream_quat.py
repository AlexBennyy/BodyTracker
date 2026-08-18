import asyncio
import csv
import json
import time

import websockets

# --- Config ---
CSV_PATH = "example_imu_two_sensors_curl_50hz.csv"
HOST = "127.0.0.1"
PORT = 8765

SENSOR_ID = "forearm"
PLAYBACK_SPEED = 1.0

REQUIRED = {"t_sec", "sensor_id", "quat_w", "quat_x", "quat_y", "quat_z"}


def load_rows(path: str, sensor_id: str):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")

        for r in reader:
            if r["sensor_id"] != sensor_id:
                continue
            rows.append({
                "t_sec": float(r["t_sec"]),
                "quat": [float(r["quat_w"]), float(r["quat_x"]), float(r["quat_y"]), float(r["quat_z"])],
            })

    if not rows:
        raise ValueError(f"No rows found for sensor_id='{sensor_id}'")

    rows.sort(key=lambda x: x["t_sec"])
    return rows


async def stream_data(websocket, rows):
    t0_wall = time.time()
    t0_data = rows[0]["t_sec"]

    for r in rows:
        t_data = r["t_sec"]
        target_wall = t0_wall + (t_data - t0_data) / max(PLAYBACK_SPEED, 1e-6)
        sleep_for = target_wall - time.time()
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)

        msg = {"t_sec": t_data, "sensor_id": SENSOR_ID, "quat": r["quat"]}
        await websocket.send(json.dumps(msg))

    # loop forever so refreshing the page works (optional)
    while True:
        await asyncio.sleep(0.25)


async def handler(websocket):
    print("Client connected")
    rows = load_rows(CSV_PATH, SENSOR_ID)
    await stream_data(websocket, rows)


async def main():
    print(f"Streaming {CSV_PATH} (sensor_id={SENSOR_ID}) on ws://{HOST}:{PORT}")
    async with websockets.serve(handler, HOST, PORT):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())

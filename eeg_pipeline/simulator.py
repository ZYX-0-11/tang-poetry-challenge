"""WebSocket simulator: reads sim_data.csv and pushes one JSON row per second.

Usage:
    python simulator.py [--port 9999] [--csv sim_data.csv]

Connect the frontend to ws://localhost:9999 and it will receive:
    {"state": 0|1, "tbr": 12.345}
"""

from __future__ import annotations

import asyncio
import csv
import json
import signal
from pathlib import Path

import websockets
from websockets.asyncio.server import serve

ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "sim_data.csv"


def load_rows(csv_path: Path) -> list[dict]:
    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "state": int(row["state"]),
                "tbr": float(row["tbr"]),
            })
    return rows


async def stream_rows(websocket, rows: list[dict], interval: float = 1.0):
    """Push rows one-by-one, looping forever."""
    idx = 0
    total = len(rows)
    print(f"Client connected, streaming {total} rows at {interval}s intervals")
    try:
        while True:
            row = rows[idx]
            msg = json.dumps(row)
            await websocket.send(msg)
            idx = (idx + 1) % total
            if idx == 0:
                print(f"  ...looped back to start (total {total} rows)")
            await asyncio.sleep(interval)
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")


async def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--csv", type=str, default=str(DEFAULT_CSV))
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rows = load_rows(csv_path)
    print(f"Loaded {len(rows)} rows from {csv_path}")
    print(f"State=0 count: {sum(1 for r in rows if r['state']==0)}")
    print(f"State=1 count: {sum(1 for r in rows if r['state']==1)}")

    async def handler(websocket):
        await stream_rows(websocket, rows, args.interval)

    stop = asyncio.get_running_loop().create_future()

    async with serve(handler, "localhost", args.port) as server:
        print(f"WebSocket server listening on ws://localhost:{args.port}")
        print("Waiting for EEG frontend to connect... (Ctrl+C to stop)")

        def shutdown():
            if not stop.done():
                stop.set_result(None)

        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT, shutdown)
        loop.add_signal_handler(signal.SIGTERM, shutdown)

        await stop

    print("Shutting down.")


if __name__ == "__main__":
    asyncio.run(main())

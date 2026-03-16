from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import psutil

from ml_running import CSVRunLogger, SystemCollector


def _timestamp_name() -> str:
    now = dt.datetime.now(dt.timezone.utc)
    return now.strftime("%Y%m%d_%H%M%S")


def run_monitor(
    *,
    duration_sec: float,
    log_dir: Path,
    log_fps: float,
    label: str,
) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    out_csv = log_dir / f"{_timestamp_name()}.csv"

    proc = psutil.Process()
    collectors = [SystemCollector(proc)]
    logger = CSVRunLogger(out_csv=out_csv, fps=float(log_fps), collectors=collectors)

    logger.start()
    logger.enable()
    logger.update_state(phase="monitor", train_active=0)
    logger.mark_event(json.dumps({"monitor_label": label}))

    time.sleep(duration_sec)

    logger.disable()
    logger.stop()
    return out_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--duration-sec", type=float, default=300.0, help="Monitoring duration in seconds")
    p.add_argument("--log-dir", type=str, default="logs_monitor", help="Output directory for CSV logs")
    p.add_argument("--log-fps", type=float, default=1.0)
    p.add_argument("--label", type=str, default="idle", help="Label stored in event row")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = run_monitor(
        duration_sec=float(args.duration_sec),
        log_dir=Path(args.log_dir),
        log_fps=float(args.log_fps),
        label=args.label,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

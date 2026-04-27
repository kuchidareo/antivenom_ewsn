from __future__ import annotations

from pathlib import Path
import csv
import json
from typing import Any

import pandas as pd


def _read_rows(csv_path: Path) -> list[dict[str, Any]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _extract_run_info(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        event = (row.get("event") or "").strip()
        if not event.startswith("{"):
            continue
        try:
            payload = json.loads(event)
        except json.JSONDecodeError:
            continue
        run_info = payload.get("run_info")
        if isinstance(run_info, dict):
            return run_info
    raise ValueError(f"Could not find run_info event in CSV with {len(rows)} rows")


def _has_eval(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        event = (row.get("event") or "").strip()
        if event.startswith("{\"eval\""):
            return True
    return False


def _pick_csv(csv_paths: list[Path]) -> Path:
    if not csv_paths:
        raise FileNotFoundError("No CSV files found")

    best_path: Path | None = None
    best_key: tuple[int, int, str] | None = None
    for path in sorted(csv_paths):
        rows = _read_rows(path)
        key = (1 if _has_eval(rows) else 0, len(rows), path.name)
        if best_key is None or key > best_key:
            best_key = key
            best_path = path

    if best_path is None:
        raise FileNotFoundError("Failed to select a CSV file")
    return best_path


def _load_one(csv_path: Path, run_label: str, device_id: str) -> pd.DataFrame:
    rows = _read_rows(csv_path)
    run_info = _extract_run_info(rows)
    df = pd.DataFrame(rows)

    df["source_label"] = run_label
    df["device_id"] = device_id
    df["poisoning_type"] = str(run_info.get("poison_type", run_label))
    df["poison_frac"] = pd.to_numeric(run_info.get("poison_frac"), errors="coerce")
    df["run_csv"] = str(csv_path.resolve())
    df["dataset"] = str(run_info.get("dataset", ""))
    df["model"] = str(run_info.get("model", ""))

    if "ts_unix" in df.columns:
        df["ts_unix"] = pd.to_numeric(df["ts_unix"], errors="coerce")
    if "epoch" in df.columns:
        df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")

    return df


def load_logs(root_dir: str | Path) -> pd.DataFrame:
    root = Path(root_dir).resolve()
    device_id = root.name

    sources = {
        "baseclean": root.parent / "baseclean",
        "clean": root / "clean",
        "blurring": root / "blurring",
    }

    frames: list[pd.DataFrame] = []
    for run_label, directory in sources.items():
        csv_paths = sorted(directory.glob("*.csv"))
        if not csv_paths:
            raise FileNotFoundError(f"No CSV files found under {directory}")
        csv_path = _pick_csv(csv_paths)
        frames.append(_load_one(csv_path, run_label=run_label, device_id=device_id))

    return pd.concat(frames, ignore_index=True)

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


def parse_run_spec(spec: str) -> tuple[str, str, Path]:
    if "=" not in spec:
        raise ValueError(
            f"Invalid run spec '{spec}'. Expected 'group:label=/path/to.csv' or 'label=/path/to.csv'."
        )
    left, right = spec.split("=", 1)
    path = Path(right).expanduser().resolve()
    left = left.strip()
    if ":" in left:
        group, label = left.split(":", 1)
    else:
        group = left
        label = left
    group = group.strip()
    label = label.strip()
    if not group or not label:
        raise ValueError(f"Invalid run spec '{spec}'. Group and label must be non-empty.")
    return group, label, path


def _load_one(csv_path: Path, source_group: str, source_label: str, device_id: str) -> pd.DataFrame:
    rows = _read_rows(csv_path)
    run_info = _extract_run_info(rows)
    df = pd.DataFrame(rows)

    df["source_group"] = source_group
    df["source_label"] = source_label
    df["device_id"] = device_id
    df["poisoning_type"] = str(run_info.get("poison_type", source_label))
    if df["poisoning_type"].iloc[0] == "none":
        df["poisoning_type"] = "clean"
    df["poison_frac"] = pd.to_numeric(run_info.get("poison_frac"), errors="coerce")
    df["ood_frac"] = pd.to_numeric(run_info.get("ood_frac"), errors="coerce")
    df["run_csv"] = str(csv_path)
    df["dataset"] = str(run_info.get("dataset", ""))
    df["model"] = str(run_info.get("model", ""))

    if "ts_unix" in df.columns:
        df["ts_unix"] = pd.to_numeric(df["ts_unix"], errors="coerce")
    if "epoch" in df.columns:
        df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    if "mem_percent" in df.columns:
        df["mem_percent"] = pd.to_numeric(df["mem_percent"], errors="coerce")

    return df


def load_logs(
    reference_spec: str,
    run_specs: list[str],
    device_id: str = "scenario_device",
) -> pd.DataFrame:
    specs = [reference_spec] + list(run_specs)
    seen: set[tuple[str, str]] = set()
    frames: list[pd.DataFrame] = []

    for spec in specs:
        source_group, source_label, csv_path = parse_run_spec(spec)
        key = (source_group, source_label)
        if key in seen:
            raise ValueError(f"Duplicate source group/label pair: {source_group}:{source_label}")
        seen.add(key)
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing CSV: {csv_path}")
        frames.append(
            _load_one(
                csv_path=csv_path,
                source_group=source_group,
                source_label=source_label,
                device_id=device_id,
            )
        )

    return pd.concat(frames, ignore_index=True)

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def step_changes(rows: List[Dict[str, str]]) -> List[float]:
    ordered = sorted(rows, key=lambda row: int(row["global_step"]))
    values = [float(row["rmse"]) for row in ordered]
    return [abs(curr - prev) for prev, curr in zip(values, values[1:])]


def calculate_layer_smoothness(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    by_condition_layer: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_condition_layer[(row["condition"], row["layer"])].append(row)

    out_rows: List[Dict[str, object]] = []
    for (condition, layer), layer_rows in sorted(by_condition_layer.items()):
        changes = step_changes(layer_rows)
        out_rows.append(
            {
                "condition": condition,
                "layer": layer,
                "num_steps": len(layer_rows),
                "num_changes": len(changes),
                "mean_abs_step_change": mean(changes) if changes else 0.0,
                "median_abs_step_change": median(changes) if changes else 0.0,
                "max_abs_step_change": max(changes) if changes else 0.0,
            }
        )
    return out_rows


def print_rows(rows: List[Dict[str, object]], top_k: int) -> None:
    by_layer: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_layer[str(row["layer"])].append(row)

    for layer, layer_rows in sorted(by_layer.items()):
        print(f"\n[{layer}]")
        ranked = sorted(layer_rows, key=lambda row: float(row["mean_abs_step_change"]), reverse=True)
        for row in ranked[:top_k]:
            print(
                f"{row['condition']}: "
                f"mean_abs_step_change={float(row['mean_abs_step_change']):.10g}, "
                f"median={float(row['median_abs_step_change']):.10g}, "
                f"max={float(row['max_abs_step_change']):.10g}, "
                f"steps={row['num_steps']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate layer-wise RMSE smoothness as mean absolute step-to-step change."
    )
    parser.add_argument(
        "--layer-csv",
        type=Path,
        default=SCRIPT_DIR / "rmse_layer_batch_timeseries.csv",
        help="Input CSV produced by rmse_batch_tracking_analysis.py.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=SCRIPT_DIR / "rmse_layer_smoothness.csv",
        help="Output CSV for layer-wise smoothness metrics.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="Number of highest-change layers to print per condition.",
    )
    args = parser.parse_args()

    rows = read_csv(args.layer_csv)
    smoothness_rows = calculate_layer_smoothness(rows)
    write_csv(args.output_csv, smoothness_rows)
    print_rows(smoothness_rows, args.top_k)
    print(f"\nsaved={args.output_csv}")


if __name__ == "__main__":
    main()

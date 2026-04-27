from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Optional, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
RMSE_ANALYSIS_DIR = ROOT / "1_gradient_RMSE_analysis"
sys.path.insert(0, str(RMSE_ANALYSIS_DIR))

from gradient_step_analysis import list_pt_files, load_grads, squared_error_sum  # noqa: E402


REF_CONDITION = "clean"
DEFAULT_CONDITIONS = [
    "clean",
    "augmentation",
    "ood",
    "blurring",
    "label-flip",
    "steganography",
    "occlusion",
]
CONDITION_ALIASES = {
    "augmentation": ["augmentation", "data_augmentation"],
    "data_augmentation": ["data_augmentation", "augmentation"],
}
STEP_RE = re.compile(r"epoch_(?P<epoch>\d+)_step_(?P<step>\d+)\.pt$")


def candidate_condition_dirs(root: Path, condition: str) -> List[Path]:
    names = CONDITION_ALIASES.get(condition, [condition])
    return [root / name for name in names]


def latest_run_dir(condition_dir: Path) -> Optional[Path]:
    if not condition_dir.exists() or not condition_dir.is_dir():
        return None
    run_dirs = [
        path
        for path in condition_dir.iterdir()
        if path.is_dir() and any(path.glob("*.pt"))
    ]
    if not run_dirs:
        return None
    return sorted(run_dirs)[-1]


def resolve_run_dir(root: Path, condition: str) -> Optional[Path]:
    for condition_dir in candidate_condition_dirs(root, condition):
        run_dir = latest_run_dir(condition_dir)
        if run_dir is not None:
            return run_dir
    return None


def parse_step_file(step_file: str) -> Tuple[int, int]:
    match = STEP_RE.match(step_file)
    if not match:
        return -1, -1
    return int(match.group("epoch")), int(match.group("step"))


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_condition(rows: List[Dict[str, object]]) -> Dict[str, object]:
    values = [float(row["rmse"]) for row in rows]
    first = values[0] if values else math.nan
    last = values[-1] if values else math.nan
    return {
        "num_steps": len(values),
        "mean_rmse": mean(values) if values else math.nan,
        "median_rmse": median(values) if values else math.nan,
        "min_rmse": min(values) if values else math.nan,
        "max_rmse": max(values) if values else math.nan,
        "first_rmse": first,
        "last_rmse": last,
        "delta_last_minus_first": last - first if values else math.nan,
    }


def collect_condition(
    condition: str,
    ref_dir: Path,
    target_dir: Path,
    requested_layers: Optional[List[str]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    out_step_rows: List[Dict[str, object]] = []
    out_layer_rows: List[Dict[str, object]] = []

    ref_files = list_pt_files(ref_dir)
    target_files = list_pt_files(target_dir)
    common_names = sorted(set(ref_files) & set(target_files))
    if not common_names:
        raise ValueError(f"No matching .pt filenames between {ref_dir} and {target_dir}")

    for step_file in common_names:
        ref_grads = load_grads(ref_files[step_file])
        target_grads = load_grads(target_files[step_file])
        common_layers = sorted(set(ref_grads) & set(target_grads))
        layers = requested_layers if requested_layers else common_layers
        missing = [name for name in layers if name not in ref_grads or name not in target_grads]
        if missing:
            raise KeyError(f"Requested layers missing from one side for {step_file}: {missing}")

        epoch, global_step = parse_step_file(step_file)
        total_squared_error = 0.0
        total_values = 0

        for layer_name in layers:
            squared_error, num_values = squared_error_sum(ref_grads[layer_name], target_grads[layer_name])
            total_squared_error += squared_error
            total_values += num_values
            out_layer_rows.append(
                {
                    "condition": condition,
                    "ref_run": str(ref_dir),
                    "target_run": str(target_dir),
                    "step_file": step_file,
                    "epoch": epoch,
                    "global_step": global_step,
                    "layer": layer_name,
                    "rmse": math.sqrt(squared_error / num_values) if num_values else math.nan,
                    "num_values": num_values,
                }
            )

        out_step_rows.append(
            {
                "condition": condition,
                "ref_run": str(ref_dir),
                "target_run": str(target_dir),
                "step_file": step_file,
                "epoch": epoch,
                "global_step": global_step,
                "rmse": math.sqrt(total_squared_error / total_values) if total_values else math.nan,
                "num_values": total_values,
                "num_layers": len(layers),
            }
        )

    out_step_rows.sort(key=lambda row: (int(row["global_step"]), int(row["epoch"]), str(row["step_file"])))
    out_layer_rows.sort(key=lambda row: (str(row["layer"]), int(row["global_step"]), int(row["epoch"]), str(row["step_file"])))
    return out_step_rows, out_layer_rows


def parse_run_overrides(values: Iterable[str]) -> Dict[str, Path]:
    runs: Dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected CONDITION=PATH, got {value}")
        condition, path = value.split("=", 1)
        runs[condition.strip()] = Path(path).expanduser()
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Track clean-vs-condition gradient RMSE across batches.")
    parser.add_argument("--root", type=Path, default=ROOT, help="Directory containing condition run folders.")
    parser.add_argument("--ref-condition", type=str, default=REF_CONDITION)
    parser.add_argument("--conditions", type=str, nargs="*", default=DEFAULT_CONDITIONS)
    parser.add_argument("--runs", type=str, nargs="*", default=[], help="Explicit CONDITION=RUN_DIR overrides.")
    parser.add_argument("--layers", type=str, nargs="*", default=None, help="Optional parameter names to include.")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    run_overrides = parse_run_overrides(args.runs)

    ref_dir = run_overrides.get(args.ref_condition) or resolve_run_dir(root, args.ref_condition)
    if ref_dir is None:
        raise FileNotFoundError(f"No reference run found for {args.ref_condition}")
    ref_dir = ref_dir.resolve()

    all_step_rows: List[Dict[str, object]] = []
    all_layer_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    print(f"reference_condition={args.ref_condition}")
    print(f"reference_run={ref_dir}")

    for condition in args.conditions:
        target_dir = run_overrides.get(condition)
        if target_dir is None:
            target_dir = ref_dir if condition == args.ref_condition else resolve_run_dir(root, condition)
        if target_dir is None:
            aliases = ", ".join(str(path.relative_to(root)) for path in candidate_condition_dirs(root, condition))
            print(f"skipped {condition}: no .pt run found under {aliases}")
            continue
        target_dir = target_dir.resolve()

        step_rows, layer_rows = collect_condition(
            condition=condition,
            ref_dir=ref_dir,
            target_dir=target_dir,
            requested_layers=args.layers,
        )
        all_step_rows.extend(step_rows)
        all_layer_rows.extend(layer_rows)
        summary = summarize_condition(step_rows)
        summary_rows.append(
            {
                "condition": condition,
                "ref_run": str(ref_dir),
                "target_run": str(target_dir),
                **summary,
            }
        )
        print(
            f"{condition}: target={target_dir}, "
            f"steps={summary['num_steps']}, mean_rmse={summary['mean_rmse']:.6g}, "
            f"last_rmse={summary['last_rmse']:.6g}"
        )

    write_csv(output_dir / "rmse_batch_timeseries.csv", all_step_rows)
    write_csv(output_dir / "rmse_layer_batch_timeseries.csv", all_layer_rows)
    write_csv(output_dir / "rmse_batch_summary.csv", summary_rows)
    print(f"saved={output_dir / 'rmse_batch_timeseries.csv'}")
    print(f"saved={output_dir / 'rmse_layer_batch_timeseries.csv'}")
    print(f"saved={output_dir / 'rmse_batch_summary.csv'}")


if __name__ == "__main__":
    main()

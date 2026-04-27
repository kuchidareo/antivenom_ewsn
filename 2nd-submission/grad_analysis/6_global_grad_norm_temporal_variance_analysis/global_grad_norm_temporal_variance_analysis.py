from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch


DEFAULT_CONDITIONS = [
    "clean",
    "clean_ref",
    "ood",
    "augmentation",
    "blurring",
    "label-flip",
    "steganography",
    "occlusion",
]
CONDITION_ALIASES = {
    "augmentation": ["augmentation", "data_augmentation"],
    "data_augmentation": ["data_augmentation", "augmentation"],
}
OUTPUT_FIELDS = [
    "condition",
    "epoch",
    "step",
    "global_step",
    "step_file",
    "scope",
    "layer",
    "batch_loss",
    "global_grad_norm",
    "global_grad_energy",
    "global_grad_norm_mean",
    "global_grad_norm_variance",
    "global_grad_norm_std",
    "window_size",
    "window_count",
    "variance_scope",
    "num_grad_values",
]
STEP_RE = re.compile(r"^epoch_(?P<epoch>\d+)_step_(?P<step>\d+)\.pt$")


def canonical_condition(condition: str) -> str:
    condition = condition.strip()
    if condition == "data_augmentation":
        return "augmentation"
    return condition


def candidate_condition_dirs(root: Path, condition: str) -> List[Path]:
    names = CONDITION_ALIASES.get(condition, [condition])
    return [root / name for name in names]


def list_run_dirs(condition_dir: Path) -> List[Path]:
    if not condition_dir.exists() or not condition_dir.is_dir():
        return []
    return [
        path
        for path in condition_dir.iterdir()
        if path.is_dir() and any(path.glob("epoch_*_step_*.pt"))
    ]


def latest_run_dir(condition_dir: Path) -> Optional[Path]:
    run_dirs = list_run_dirs(condition_dir)
    if not run_dirs:
        return None
    return sorted(run_dirs)[-1]


def resolve_run_dir(root: Path, condition: str) -> Optional[Path]:
    for condition_dir in candidate_condition_dirs(root, condition):
        run_dir = latest_run_dir(condition_dir)
        if run_dir is not None:
            return run_dir
    return None


def parse_step_file(path: Path) -> Dict[str, int]:
    match = STEP_RE.match(path.name)
    if not match:
        raise ValueError(f"Unexpected gradient filename: {path.name}")
    return {
        "epoch": int(match.group("epoch")),
        "global_step": int(match.group("step")),
    }


def parse_float(value: object) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def read_csv_rows(path: Path) -> List[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def run_info_from_log(path: Path) -> Dict[str, object]:
    for row in read_csv_rows(path):
        event = str(row.get("event", ""))
        if "run_info" not in event:
            continue
        try:
            payload = json.loads(event)
        except json.JSONDecodeError:
            continue
        run_info = payload.get("run_info")
        if isinstance(run_info, dict):
            return run_info
    return {}


def log_matches_run(log_csv: Path, grad_dir: Path, condition: Optional[str] = None) -> bool:
    run_info = run_info_from_log(log_csv)
    grad_log_dir = str(run_info.get("grad_log_dir", ""))
    if grad_log_dir and Path(grad_log_dir).name == grad_dir.name:
        return True
    if condition is not None:
        poison_type = canonical_condition(str(run_info.get("poison_type", "")))
        if poison_type and poison_type != canonical_condition(condition):
            return False
    return log_csv.stem == grad_dir.name


def find_log_csv(log_dirs: Sequence[Path], grad_dir: Path, condition: Optional[str] = None) -> Optional[Path]:
    candidates: List[Path] = []
    for log_dir in log_dirs:
        if not log_dir.exists():
            continue
        candidates.extend(sorted(log_dir.glob("*.csv")))
    for path in candidates:
        if path.name.endswith("_grad_analysis.csv"):
            continue
        if log_matches_run(path, grad_dir, condition=condition):
            return path
    return None


def load_batch_losses(log_csv: Optional[Path]) -> Dict[int, float]:
    if log_csv is None:
        return {}

    losses: Dict[int, float] = {}
    for row in read_csv_rows(log_csv):
        if row.get("event") != "batch_end":
            continue
        step_value = parse_float(row.get("global_step"))
        loss_value = parse_float(row.get("batch_loss") or row.get("loss"))
        if step_value is None or loss_value is None:
            continue
        losses[int(step_value)] = loss_value
    return losses


def list_grad_files(grad_dir: Path) -> List[Path]:
    if not grad_dir.exists():
        raise FileNotFoundError(f"Gradient directory not found: {grad_dir}")
    if not grad_dir.is_dir():
        raise NotADirectoryError(f"Expected directory: {grad_dir}")
    files = sorted(grad_dir.glob("epoch_*_step_*.pt"))
    if not files:
        raise ValueError(f"No gradient .pt files found in {grad_dir}")
    return files


def load_grads(pt_file: Path) -> Dict[str, torch.Tensor]:
    grads = torch.load(pt_file, map_location="cpu")
    if not isinstance(grads, dict):
        raise TypeError(f"Expected dict in {pt_file}, got {type(grads)}")
    return {str(name): value for name, value in grads.items() if isinstance(value, torch.Tensor)}


def global_grad_norm_stats(grads: Dict[str, torch.Tensor]) -> Dict[str, float]:
    energy = 0.0
    num_values = 0
    for grad in grads.values():
        values = grad.detach().cpu().float().reshape(-1)
        energy += float(torch.sum(values * values).item())
        num_values += int(values.numel())
    return {
        "global_grad_energy": energy,
        "global_grad_norm": energy ** 0.5,
        "num_grad_values": num_values,
    }


def layer_key(name: str) -> str:
    return name.rsplit(".", 1)[0] if "." in name else name


def layer_grad_norm_stats(grads: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
    energy_by_layer: Dict[str, float] = {}
    num_values_by_layer: Dict[str, int] = {}
    for name, grad in grads.items():
        layer = layer_key(name)
        values = grad.detach().cpu().float().reshape(-1)
        energy = float(torch.sum(values * values).item())
        energy_by_layer[layer] = energy_by_layer.get(layer, 0.0) + energy
        num_values_by_layer[layer] = num_values_by_layer.get(layer, 0) + int(values.numel())

    return {
        layer: {
            "global_grad_energy": energy,
            "global_grad_norm": energy ** 0.5,
            "num_grad_values": num_values_by_layer[layer],
        }
        for layer, energy in sorted(energy_by_layer.items())
    }


def population_variance(values: Sequence[float]) -> float:
    if not values:
        return math.nan
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def add_temporal_variance(rows: List[Dict[str, object]], window_size: int, min_window: int) -> List[Dict[str, object]]:
    sorted_rows: List[Dict[str, object]] = []
    grouped: Dict[tuple[str, str], List[Dict[str, object]]] = {}
    for row in rows:
        key = (str(row.get("scope", "")), str(row.get("layer", "")))
        grouped.setdefault(key, []).append(row)

    for _, group_rows in sorted(grouped.items()):
        group_rows = sorted(group_rows, key=lambda row: (int(row["epoch"]), int(row["global_step"])))
        norms = [float(row["global_grad_norm"]) for row in group_rows]

        for index, row in enumerate(group_rows):
            start = max(0, index - window_size + 1)
            window = norms[start : index + 1]
            if len(window) < min_window:
                variance = math.nan
                mean = math.nan
                std = math.nan
            else:
                mean = sum(window) / len(window)
                variance = population_variance(window)
                std = variance ** 0.5 if not math.isnan(variance) else math.nan

            row["global_grad_norm_mean"] = mean
            row["global_grad_norm_variance"] = variance
            row["global_grad_norm_std"] = std
            row["window_size"] = window_size
            row["window_count"] = len(window)
            row["variance_scope"] = "rolling_global_grad_norm"
            sorted_rows.append(row)

    return sorted(sorted_rows, key=lambda row: (int(row["global_step"]), str(row.get("scope", "")), str(row.get("layer", ""))))


def analyze_run(
    grad_dir: Path,
    log_csv: Optional[Path],
    condition: str,
    window_size: int,
    min_window: int,
) -> List[Dict[str, object]]:
    batch_losses = load_batch_losses(log_csv)
    rows: List[Dict[str, object]] = []

    for pt_file in list_grad_files(grad_dir):
        step_info = parse_step_file(pt_file)
        epoch = int(step_info["epoch"])
        global_step = int(step_info["global_step"])
        batch_loss = batch_losses.get(global_step, math.nan)
        grads = load_grads(pt_file)
        stats = global_grad_norm_stats(grads)

        rows.append(
            {
                "condition": canonical_condition(condition),
                "epoch": epoch,
                "step": global_step,
                "global_step": global_step,
                "step_file": pt_file.name,
                "scope": "global",
                "layer": "",
                "batch_loss": batch_loss,
                **stats,
            }
        )

        for layer, layer_stats in layer_grad_norm_stats(grads).items():
            rows.append(
                {
                    "condition": canonical_condition(condition),
                    "epoch": epoch,
                    "step": global_step,
                    "global_step": global_step,
                    "step_file": pt_file.name,
                    "scope": "layer",
                    "layer": layer,
                    "batch_loss": batch_loss,
                    **layer_stats,
                }
            )

    return add_temporal_variance(rows, window_size=window_size, min_window=min_window)


def write_rows(rows: Sequence[Dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in OUTPUT_FIELDS})


def analyze_conditions(
    root: Path,
    log_dirs: Sequence[Path],
    conditions: Sequence[str],
    window_size: int,
    min_window: int,
) -> List[Dict[str, object]]:
    all_rows: List[Dict[str, object]] = []
    for condition in conditions:
        grad_dir = resolve_run_dir(root, condition)
        if grad_dir is None:
            aliases = ", ".join(str(path.relative_to(root)) for path in candidate_condition_dirs(root, condition))
            print(f"skipped {condition}: no gradient run found under {aliases}")
            continue

        log_csv = find_log_csv(log_dirs, grad_dir, condition=condition)
        print(f"{canonical_condition(condition)}: grad_dir={grad_dir} log_csv={log_csv or 'none'}")
        all_rows.extend(
            analyze_run(
                grad_dir=grad_dir,
                log_csv=log_csv,
                condition=condition,
                window_size=window_size,
                min_window=min_window,
            )
        )
    return all_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze temporal variance of per-batch global gradient norm."
    )
    parser.add_argument("--grad-dir", type=Path, default=None, help="Single gradient run directory.")
    parser.add_argument("--log-csv", type=Path, default=None, help="Training log CSV for --grad-dir.")
    parser.add_argument("--condition", type=str, default=None, help="Condition name for --grad-dir.")
    parser.add_argument("--root", type=Path, default=Path(".."), help="Root containing condition directories.")
    parser.add_argument("--log-dir", type=Path, default=Path("../logs"), help="Directory containing training logs.")
    parser.add_argument("--output-csv", type=Path, default=Path("global_grad_norm_temporal_variance.csv"))
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    parser.add_argument("--window-size", type=int, default=10, help="Rolling window size over steps.")
    parser.add_argument("--min-window", type=int, default=2, help="Minimum samples before variance is emitted.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.window_size < 2:
        raise ValueError("--window-size must be at least 2")
    if args.min_window < 2 or args.min_window > args.window_size:
        raise ValueError("--min-window must be between 2 and --window-size")

    script_dir = Path(__file__).resolve().parent
    output_csv = (script_dir / args.output_csv).resolve() if not args.output_csv.is_absolute() else args.output_csv

    if args.grad_dir is not None:
        grad_dir = (script_dir / args.grad_dir).resolve() if not args.grad_dir.is_absolute() else args.grad_dir
        log_csv = None
        if args.log_csv is not None:
            log_csv = (script_dir / args.log_csv).resolve() if not args.log_csv.is_absolute() else args.log_csv
        condition = args.condition or grad_dir.parent.name
        rows = analyze_run(
            grad_dir=grad_dir,
            log_csv=log_csv,
            condition=condition,
            window_size=args.window_size,
            min_window=args.min_window,
        )
    else:
        root = (script_dir / args.root).resolve() if not args.root.is_absolute() else args.root
        log_dir = (script_dir / args.log_dir).resolve() if not args.log_dir.is_absolute() else args.log_dir
        rows = analyze_conditions(
            root=root,
            log_dirs=[log_dir],
            conditions=args.conditions,
            window_size=args.window_size,
            min_window=args.min_window,
        )

    if not rows:
        raise ValueError("No rows were collected")
    write_rows(rows, output_csv)
    print(f"rows={len(rows)} output_csv={output_csv}")


if __name__ == "__main__":
    main()

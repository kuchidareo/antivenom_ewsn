from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

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
    "grad_variance",
    "grad_kurtosis",
    "grad_excess_kurtosis",
    "num_values",
    "grad_mean",
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


def latest_run_dir(condition_dir: Path) -> Optional[Path]:
    if not condition_dir.exists() or not condition_dir.is_dir():
        return None
    run_dirs = list_run_dirs(condition_dir)
    if not run_dirs:
        return None
    return sorted(run_dirs)[-1]


def list_run_dirs(condition_dir: Path) -> List[Path]:
    if not condition_dir.exists() or not condition_dir.is_dir():
        return []
    return [
        path
        for path in condition_dir.iterdir()
        if path.is_dir() and any(path.glob("epoch_*_step_*.pt"))
    ]


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


def run_name_from_grad_dir(grad_dir: Path) -> str:
    return grad_dir.name


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
    return log_csv.stem == run_name_from_grad_dir(grad_dir)


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


def grad_distribution_stats(grad: torch.Tensor) -> Dict[str, float]:
    values = grad.detach().cpu().float().reshape(-1)
    n_values = int(values.numel())
    if n_values == 0:
        return {
            "num_values": 0,
            "grad_mean": math.nan,
            "grad_variance": math.nan,
            "grad_kurtosis": math.nan,
            "grad_excess_kurtosis": math.nan,
        }

    mean = float(values.mean().item())
    centered = values - mean
    variance = float(torch.mean(centered * centered).item())
    if variance == 0.0:
        kurtosis = math.nan
    else:
        fourth_moment = float(torch.mean(centered ** 4).item())
        kurtosis = fourth_moment / (variance ** 2)

    return {
        "num_values": n_values,
        "grad_mean": mean,
        "grad_variance": variance,
        "grad_kurtosis": kurtosis,
        "grad_excess_kurtosis": kurtosis - 3.0 if not math.isnan(kurtosis) else math.nan,
    }


def analyze_run(
    grad_dir: Path,
    log_csv: Optional[Path],
    condition: str,
    requested_layers: Optional[Sequence[str]] = None,
) -> List[Dict[str, object]]:
    batch_losses = load_batch_losses(log_csv)
    rows: List[Dict[str, object]] = []

    for pt_file in list_grad_files(grad_dir):
        step_info = parse_step_file(pt_file)
        epoch = int(step_info["epoch"])
        global_step = int(step_info["global_step"])
        batch_loss = batch_losses.get(global_step, math.nan)

        rows.append(
            {
                "condition": canonical_condition(condition),
                "epoch": epoch,
                "step": global_step,
                "global_step": global_step,
                "step_file": pt_file.name,
                "scope": "batch",
                "layer": "",
                "batch_loss": batch_loss,
                "grad_variance": "",
                "grad_kurtosis": "",
                "grad_excess_kurtosis": "",
                "num_values": "",
                "grad_mean": "",
            }
        )

        grads = load_grads(pt_file)
        layers = list(requested_layers) if requested_layers else sorted(grads)
        missing = [layer for layer in layers if layer not in grads]
        if missing:
            raise KeyError(f"Requested layers missing from {pt_file}: {missing}")

        for layer in layers:
            stats = grad_distribution_stats(grads[layer])
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
                    **stats,
                }
            )

    return rows


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
    requested_layers: Optional[Sequence[str]] = None,
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
                requested_layers=requested_layers,
            )
        )
    return all_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze batch loss plus per-layer gradient variance and excess kurtosis."
    )
    parser.add_argument("--grad-dir", type=Path, default=None, help="Single gradient run directory.")
    parser.add_argument("--log-csv", type=Path, default=None, help="Training log CSV for --grad-dir.")
    parser.add_argument("--condition", type=str, default=None, help="Condition name for --grad-dir.")
    parser.add_argument("--root", type=Path, default=Path(".."), help="Root containing condition directories.")
    parser.add_argument("--log-dir", type=Path, default=Path("../logs"), help="Directory containing training logs.")
    parser.add_argument("--output-csv", type=Path, default=Path("batch_loss_grad_distribution.csv"))
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    parser.add_argument("--layers", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    output_csv = (script_dir / args.output_csv).resolve() if not args.output_csv.is_absolute() else args.output_csv

    if args.grad_dir is not None:
        grad_dir = (script_dir / args.grad_dir).resolve() if not args.grad_dir.is_absolute() else args.grad_dir
        log_csv = None
        if args.log_csv is not None:
            log_csv = (script_dir / args.log_csv).resolve() if not args.log_csv.is_absolute() else args.log_csv
        condition = args.condition or grad_dir.parent.name
        rows = analyze_run(grad_dir=grad_dir, log_csv=log_csv, condition=condition, requested_layers=args.layers)
    else:
        root = (script_dir / args.root).resolve() if not args.root.is_absolute() else args.root
        log_dir = (script_dir / args.log_dir).resolve() if not args.log_dir.is_absolute() else args.log_dir
        rows = analyze_conditions(
            root=root,
            log_dirs=[log_dir],
            conditions=args.conditions,
            requested_layers=args.layers,
        )

    if not rows:
        raise ValueError("No rows were collected")
    write_rows(rows, output_csv)
    print(f"rows={len(rows)} output_csv={output_csv}")


if __name__ == "__main__":
    main()

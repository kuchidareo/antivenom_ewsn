from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

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


def canonical_condition(condition: str) -> str:
    if condition == "data_augmentation":
        return "augmentation"
    return condition


def candidate_condition_dirs(root: Path, condition: str) -> List[Path]:
    names = CONDITION_ALIASES.get(condition, [condition])
    return [root / name for name in names]


def latest_run_dir(condition_dir: Path) -> Optional[Path]:
    if not condition_dir.exists() or not condition_dir.is_dir():
        return None
    run_dirs = [
        path
        for path in condition_dir.iterdir()
        if path.is_dir() and any(path.glob("epoch_*_step_*.pt"))
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


def parse_step_file(path: Path) -> Tuple[int, int]:
    match = STEP_RE.match(path.name)
    if not match:
        raise ValueError(f"Unexpected gradient filename: {path.name}")
    return int(match.group("epoch")), int(match.group("step"))


def list_grad_files(run_dir: Path) -> List[Path]:
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    files = sorted(run_dir.glob("epoch_*_step_*.pt"))
    if not files:
        raise ValueError(f"No gradient .pt files found in {run_dir}")
    return files


def load_grads(pt_file: Path) -> Dict[str, torch.Tensor]:
    loaded = torch.load(pt_file, map_location="cpu")
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected dict in {pt_file}, got {type(loaded)}")
    return {
        str(name): value
        for name, value in loaded.items()
        if isinstance(value, torch.Tensor) and is_parameter_grad_name(str(name))
    }


def is_parameter_grad_name(name: str) -> bool:
    return "." in name and not name.startswith("input")


def layer_key(name: str) -> str:
    return name.rsplit(".", 1)[0] if "." in name else name


def finite_or_empty(value: float) -> object:
    if math.isnan(value) or math.isinf(value):
        return ""
    return value


def kurtosis_stats(values: torch.Tensor) -> Dict[str, float]:
    n_values = int(values.numel())
    if n_values == 0:
        return {
            "num_values": 0,
            "mean": math.nan,
            "variance": math.nan,
            "kurtosis": math.nan,
            "excess_kurtosis": math.nan,
        }

    values = values.float()
    grad_mean = float(values.mean().item())
    centered = values - grad_mean
    variance = float(torch.mean(centered * centered).item())
    if variance == 0.0:
        kurtosis = math.nan
    else:
        fourth_moment = float(torch.mean(centered ** 4).item())
        kurtosis = fourth_moment / (variance ** 2)

    return {
        "num_values": n_values,
        "mean": grad_mean,
        "variance": variance,
        "kurtosis": kurtosis,
        "excess_kurtosis": kurtosis - 3.0 if not math.isnan(kurtosis) else math.nan,
    }


def grad_distribution_stats(grad: torch.Tensor) -> Dict[str, object]:
    values = grad.detach().cpu().float().reshape(-1)
    total_num_values = int(values.numel())
    active_values = values[values != 0]
    signed = kurtosis_stats(active_values)
    absolute = kurtosis_stats(active_values.abs())
    l2_norm = float(torch.linalg.vector_norm(values).item())
    energy = float(torch.sum(values * values).item())

    return {
        "num_active_values": signed["num_values"],
        "total_num_values": total_num_values,
        "active_ratio": signed["num_values"] / total_num_values if total_num_values else math.nan,
        "grad_mean": signed["mean"],
        "abs_grad_mean": absolute["mean"],
        "grad_variance": signed["variance"],
        "grad_kurtosis": signed["kurtosis"],
        "grad_excess_kurtosis": signed["excess_kurtosis"],
        "abs_grad_kurtosis": absolute["kurtosis"],
        "abs_grad_excess_kurtosis": absolute["excess_kurtosis"],
        "l2_norm": l2_norm,
        "energy": energy,
    }


def layer_l2_stats(grads: Dict[str, torch.Tensor]) -> Tuple[Dict[str, Dict[str, object]], float]:
    sqsum: Dict[str, float] = {}
    numel: Dict[str, int] = {}
    active: Dict[str, int] = {}

    for name, grad in grads.items():
        key = layer_key(name)
        values = grad.detach().cpu().float().reshape(-1)
        energy = float(torch.sum(values * values).item())
        sqsum[key] = sqsum.get(key, 0.0) + energy
        numel[key] = numel.get(key, 0) + int(values.numel())
        active[key] = active.get(key, 0) + int((values != 0).sum().item())

    total_energy = sum(sqsum.values()) + 1e-12
    max_energy_share = max(sqsum.values()) / total_energy if sqsum else math.nan
    stats = {
        key: {
            "l2_norm": energy ** 0.5,
            "energy": energy,
            "energy_share": energy / total_energy,
            "num_active_values": active[key],
            "total_num_values": numel[key],
            "active_ratio": active[key] / numel[key] if numel[key] else math.nan,
        }
        for key, energy in sqsum.items()
    }
    return stats, max_energy_share


def layer_kurtosis_stats(grads: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, object]]:
    values_by_layer: Dict[str, List[torch.Tensor]] = {}
    for name, grad in grads.items():
        values_by_layer.setdefault(layer_key(name), []).append(grad.detach().cpu().float().reshape(-1))

    stats: Dict[str, Dict[str, object]] = {}
    for layer, tensors in values_by_layer.items():
        values = torch.cat(tensors)
        stats[layer] = grad_distribution_stats(values)
    return stats


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_values(rows: Sequence[Dict[str, object]], group_fields: Sequence[str], metric: str) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[object, ...], List[float]] = {}
    for row in rows:
        value = row.get(metric)
        if value in ("", None):
            continue
        parsed = float(value)
        if math.isnan(parsed) or math.isinf(parsed):
            continue
        key = tuple(row[field] for field in group_fields)
        grouped.setdefault(key, []).append(parsed)

    out_rows: List[Dict[str, object]] = []
    for key, values in sorted(grouped.items()):
        summary = {field: key[index] for index, field in enumerate(group_fields)}
        summary.update(
            {
                "metric": metric,
                "num_steps": len(values),
                "mean": mean(values),
                "median": median(values),
                "min": min(values),
                "max": max(values),
                "first": values[0],
                "last": values[-1],
                "delta_last_minus_first": values[-1] - values[0],
            }
        )
        out_rows.append(summary)
    return out_rows


def analyze_run(
    condition: str,
    run_dir: Path,
    requested_layers: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    parameter_rows: List[Dict[str, object]] = []
    layer_kurtosis_rows: List[Dict[str, object]] = []
    layer_rows: List[Dict[str, object]] = []

    for pt_file in list_grad_files(run_dir):
        epoch, global_step = parse_step_file(pt_file)
        grads = load_grads(pt_file)
        if not grads:
            continue

        layers = list(requested_layers) if requested_layers else sorted(grads)
        missing = [layer for layer in layers if layer not in grads]
        if missing:
            raise KeyError(f"Requested parameter gradients missing from {pt_file}: {missing}")

        selected_grads = {layer: grads[layer] for layer in layers}
        layer_stats, max_energy_share = layer_l2_stats(selected_grads)
        layer_kurtosis = layer_kurtosis_stats(selected_grads)

        for name in layers:
            stats = grad_distribution_stats(selected_grads[name])
            parameter_rows.append(
                {
                    "condition": canonical_condition(condition),
                    "run_dir": str(run_dir),
                    "epoch": epoch,
                    "global_step": global_step,
                    "step_file": pt_file.name,
                    "parameter": name,
                    "layer": layer_key(name),
                    "num_active_values": stats["num_active_values"],
                    "total_num_values": stats["total_num_values"],
                    "active_ratio": finite_or_empty(float(stats["active_ratio"])),
                    "grad_mean": finite_or_empty(float(stats["grad_mean"])),
                    "abs_grad_mean": finite_or_empty(float(stats["abs_grad_mean"])),
                    "grad_variance": finite_or_empty(float(stats["grad_variance"])),
                    "grad_kurtosis": finite_or_empty(float(stats["grad_kurtosis"])),
                    "grad_excess_kurtosis": finite_or_empty(float(stats["grad_excess_kurtosis"])),
                    "abs_grad_kurtosis": finite_or_empty(float(stats["abs_grad_kurtosis"])),
                    "abs_grad_excess_kurtosis": finite_or_empty(float(stats["abs_grad_excess_kurtosis"])),
                    "l2_norm": stats["l2_norm"],
                    "energy": stats["energy"],
                    "max_energy_share": max_energy_share,
                }
            )

        for layer, stats in sorted(layer_kurtosis.items()):
            layer_kurtosis_rows.append(
                {
                    "condition": canonical_condition(condition),
                    "run_dir": str(run_dir),
                    "epoch": epoch,
                    "global_step": global_step,
                    "step_file": pt_file.name,
                    "layer": layer,
                    "num_active_values": stats["num_active_values"],
                    "total_num_values": stats["total_num_values"],
                    "active_ratio": finite_or_empty(float(stats["active_ratio"])),
                    "grad_mean": finite_or_empty(float(stats["grad_mean"])),
                    "abs_grad_mean": finite_or_empty(float(stats["abs_grad_mean"])),
                    "grad_variance": finite_or_empty(float(stats["grad_variance"])),
                    "grad_kurtosis": finite_or_empty(float(stats["grad_kurtosis"])),
                    "grad_excess_kurtosis": finite_or_empty(float(stats["grad_excess_kurtosis"])),
                    "abs_grad_kurtosis": finite_or_empty(float(stats["abs_grad_kurtosis"])),
                    "abs_grad_excess_kurtosis": finite_or_empty(float(stats["abs_grad_excess_kurtosis"])),
                    "max_energy_share": max_energy_share,
                }
            )

        for layer, stats in sorted(layer_stats.items()):
            layer_rows.append(
                {
                    "condition": canonical_condition(condition),
                    "run_dir": str(run_dir),
                    "epoch": epoch,
                    "global_step": global_step,
                    "step_file": pt_file.name,
                    "layer": layer,
                    "l2_norm": stats["l2_norm"],
                    "energy": stats["energy"],
                    "energy_share": stats["energy_share"],
                    "max_energy_share": max_energy_share,
                    "num_active_values": stats["num_active_values"],
                    "total_num_values": stats["total_num_values"],
                    "active_ratio": finite_or_empty(float(stats["active_ratio"])),
                }
            )

    parameter_rows.sort(key=lambda row: (str(row["condition"]), str(row["parameter"]), int(row["global_step"])))
    layer_kurtosis_rows.sort(key=lambda row: (str(row["condition"]), str(row["layer"]), int(row["global_step"])))
    layer_rows.sort(key=lambda row: (str(row["condition"]), str(row["layer"]), int(row["global_step"])))
    return parameter_rows, layer_kurtosis_rows, layer_rows


def parse_run_overrides(values: Iterable[str]) -> Dict[str, Path]:
    runs: Dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected CONDITION=PATH, got {value}")
        condition, path = value.split("=", 1)
        runs[condition.strip()] = Path(path).expanduser()
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate per-step gradient kurtosis and layer L2 norm.")
    parser.add_argument("--root", type=Path, default=ROOT, help="Directory containing condition run folders.")
    parser.add_argument("--conditions", nargs="*", default=DEFAULT_CONDITIONS)
    parser.add_argument("--runs", nargs="*", default=[], help="Explicit CONDITION=RUN_DIR overrides.")
    parser.add_argument("--layers", nargs="*", default=None, help="Optional parameter names to include.")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    run_overrides = parse_run_overrides(args.runs)

    all_parameter_rows: List[Dict[str, object]] = []
    all_layer_kurtosis_rows: List[Dict[str, object]] = []
    all_layer_rows: List[Dict[str, object]] = []

    for condition in args.conditions:
        run_dir = run_overrides.get(condition) or resolve_run_dir(root, condition)
        if run_dir is None:
            aliases = ", ".join(str(path.relative_to(root)) for path in candidate_condition_dirs(root, condition))
            print(f"skipped {condition}: no .pt run found under {aliases}")
            continue
        run_dir = run_dir.resolve()
        parameter_rows, layer_kurtosis_rows, layer_rows = analyze_run(condition, run_dir, requested_layers=args.layers)
        all_parameter_rows.extend(parameter_rows)
        all_layer_kurtosis_rows.extend(layer_kurtosis_rows)
        all_layer_rows.extend(layer_rows)
        print(
            f"{canonical_condition(condition)}: run={run_dir}, "
            f"steps={len({row['global_step'] for row in layer_rows})}, "
            f"parameter_rows={len(parameter_rows)}, layer_rows={len(layer_rows)}"
        )

    if not all_parameter_rows or not all_layer_rows:
        raise ValueError("No gradient rows were collected")

    parameter_csv = output_dir / "kurtosis_parameter_batch_timeseries.csv"
    layer_kurtosis_csv = output_dir / "kurtosis_layer_batch_timeseries.csv"
    layer_csv = output_dir / "l2_layer_batch_timeseries.csv"
    parameter_summary_csv = output_dir / "kurtosis_parameter_summary.csv"
    layer_kurtosis_summary_csv = output_dir / "kurtosis_layer_summary.csv"
    layer_summary_csv = output_dir / "l2_layer_summary.csv"

    parameter_summary_rows: List[Dict[str, object]] = []
    for metric in ("grad_excess_kurtosis", "abs_grad_excess_kurtosis", "grad_variance", "l2_norm"):
        parameter_summary_rows.extend(
            summarize_values(all_parameter_rows, group_fields=("condition", "parameter", "layer"), metric=metric)
        )

    layer_summary_rows: List[Dict[str, object]] = []
    for metric in ("l2_norm", "energy_share", "max_energy_share", "active_ratio"):
        layer_summary_rows.extend(summarize_values(all_layer_rows, group_fields=("condition", "layer"), metric=metric))

    layer_kurtosis_summary_rows: List[Dict[str, object]] = []
    for metric in ("grad_excess_kurtosis", "abs_grad_excess_kurtosis", "grad_variance"):
        layer_kurtosis_summary_rows.extend(
            summarize_values(all_layer_kurtosis_rows, group_fields=("condition", "layer"), metric=metric)
        )

    write_csv(parameter_csv, all_parameter_rows)
    write_csv(layer_kurtosis_csv, all_layer_kurtosis_rows)
    write_csv(layer_csv, all_layer_rows)
    write_csv(parameter_summary_csv, parameter_summary_rows)
    write_csv(layer_kurtosis_summary_csv, layer_kurtosis_summary_rows)
    write_csv(layer_summary_csv, layer_summary_rows)
    print(f"saved={parameter_csv}")
    print(f"saved={layer_kurtosis_csv}")
    print(f"saved={layer_csv}")
    print(f"saved={parameter_summary_csv}")
    print(f"saved={layer_kurtosis_summary_csv}")
    print(f"saved={layer_summary_csv}")


if __name__ == "__main__":
    main()

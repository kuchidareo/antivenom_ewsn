from __future__ import annotations

import argparse
import csv
import math
import os
import re
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
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
DISPLAY_NAMES = {
    "clean": "clean",
    "augmentation": "augmentation",
    "data_augmentation": "augmentation",
    "ood": "ood",
    "blurring": "blurring",
    "label-flip": "label-flip",
    "steganography": "steganography",
    "occlusion": "occlusion",
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


def parse_step_name(name: str) -> Tuple[int, int]:
    match = STEP_RE.match(name)
    if not match:
        raise ValueError(f"Unexpected gradient filename: {name}")
    return int(match.group("epoch")), int(match.group("step"))


def list_step_files(run_dir: Path, epochs: Optional[Sequence[int]]) -> Dict[str, Path]:
    wanted = set(int(epoch) for epoch in epochs) if epochs else None
    files: Dict[str, Path] = {}
    for path in run_dir.glob("epoch_*_step_*.pt"):
        epoch, _ = parse_step_name(path.name)
        if wanted is None or epoch in wanted:
            files[path.name] = path
    return files


def is_parameter_grad_name(name: str) -> bool:
    return "." in name and not name.startswith("input")


def load_parameter_grads(pt_file: Path) -> Dict[str, torch.Tensor]:
    loaded = torch.load(pt_file, map_location="cpu")
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected dict in {pt_file}, got {type(loaded)}")
    grads: Dict[str, torch.Tensor] = {}
    for name, value in loaded.items():
        key = str(name)
        if isinstance(value, torch.Tensor) and is_parameter_grad_name(key):
            grads[key] = value.detach().cpu().float()
    return grads


def layer_key(parameter_name: str) -> str:
    return parameter_name.rsplit(".", 1)[0]


def layer_vectors_from_grads(grads: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    by_layer: Dict[str, List[torch.Tensor]] = {}
    for name in sorted(grads):
        by_layer.setdefault(layer_key(name), []).append(grads[name].reshape(-1))
    return {layer: torch.cat(values) for layer, values in by_layer.items()}


def cosine_similarity(ref_values: torch.Tensor, target_values: torch.Tensor) -> float:
    ref_values = ref_values.reshape(-1).float()
    target_values = target_values.reshape(-1).float()
    if ref_values.numel() != target_values.numel():
        raise ValueError(f"Shape mismatch: {tuple(ref_values.shape)} vs {tuple(target_values.shape)}")
    ref_norm = float(torch.linalg.vector_norm(ref_values).item())
    target_norm = float(torch.linalg.vector_norm(target_values).item())
    if ref_norm == 0.0 or target_norm == 0.0:
        return math.nan
    dot_product = float(torch.dot(ref_values, target_values).item())
    return dot_product / (ref_norm * target_norm)


def finite_or_empty(value: float) -> object:
    if math.isnan(value) or math.isinf(value):
        return ""
    return value


def safe_filename(name: str) -> str:
    return name.replace(".", "_").replace("-", "_").replace("/", "_").replace(" ", "_")


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_run_overrides(values: Iterable[str]) -> Dict[str, Path]:
    runs: Dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected CONDITION=PATH, got {value}")
        condition, path = value.split("=", 1)
        runs[canonical_condition(condition.strip())] = Path(path).expanduser()
    return runs


def ordered_conditions(present: Sequence[str], requested: Sequence[str]) -> List[str]:
    present_ordered = list(dict.fromkeys(canonical_condition(condition) for condition in present))
    requested_ordered = [canonical_condition(condition) for condition in requested]
    ordered = [condition for condition in requested_ordered if condition in present_ordered]
    ordered.extend(condition for condition in present_ordered if condition not in ordered)
    return ordered


def collect_rows(
    root: Path,
    ref_condition: str,
    requested_conditions: Sequence[str],
    run_overrides: Dict[str, Path],
    epochs: Optional[Sequence[int]],
) -> Tuple[List[Dict[str, object]], Dict[str, Path], List[str]]:
    ref_condition = canonical_condition(ref_condition)
    conditions = list(dict.fromkeys(canonical_condition(condition) for condition in requested_conditions))
    if ref_condition not in conditions:
        conditions = [ref_condition] + conditions

    run_dirs: Dict[str, Path] = {}
    for condition in conditions:
        run_dir = run_overrides.get(condition) or resolve_run_dir(root, condition)
        if run_dir is None:
            aliases = ", ".join(str(path.relative_to(root)) for path in candidate_condition_dirs(root, condition))
            print(f"skipped {condition}: no .pt run found under {aliases}")
            continue
        run_dirs[condition] = run_dir.resolve()

    if ref_condition not in run_dirs:
        raise FileNotFoundError(f"No reference run found for {ref_condition}")

    step_files_by_condition = {
        condition: list_step_files(run_dir, epochs) for condition, run_dir in run_dirs.items()
    }
    common_step_names = sorted(
        set.intersection(*(set(files.keys()) for files in step_files_by_condition.values())),
        key=parse_step_name,
    )
    if not common_step_names:
        raise ValueError("No common step files found across selected conditions/epochs.")

    rows: List[Dict[str, object]] = []
    condition_order = ordered_conditions(list(run_dirs), conditions)
    for step_name in common_step_names:
        epoch, global_step = parse_step_name(step_name)
        ref_grads = load_parameter_grads(step_files_by_condition[ref_condition][step_name])
        ref_layers = layer_vectors_from_grads(ref_grads)
        for condition in condition_order:
            target_grads = load_parameter_grads(step_files_by_condition[condition][step_name])
            target_layers = layer_vectors_from_grads(target_grads)
            common_layers = sorted(set(ref_layers) & set(target_layers))
            for layer in common_layers:
                rows.append(
                    {
                        "condition": condition,
                        "ref_condition": ref_condition,
                        "ref_run": str(run_dirs[ref_condition]),
                        "target_run": str(run_dirs[condition]),
                        "epoch": epoch,
                        "global_step": global_step,
                        "step_file": step_name,
                        "layer": layer,
                        "cosine_similarity_from_clean": finite_or_empty(
                            cosine_similarity(ref_layers[layer], target_layers[layer])
                        ),
                        "ref_l2_norm": float(torch.linalg.vector_norm(ref_layers[layer]).item()),
                        "target_l2_norm": float(torch.linalg.vector_norm(target_layers[layer]).item()),
                    }
                )

    rows.sort(key=lambda row: (str(row["layer"]), str(row["condition"]), int(row["global_step"])))
    return rows, run_dirs, common_step_names


def parse_float(value: object) -> Optional[float]:
    if value in ("", None):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def summarize_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[object, object], List[float]] = {}
    for row in rows:
        value = parse_float(row.get("cosine_similarity_from_clean"))
        if value is None:
            continue
        grouped.setdefault((row["condition"], row["layer"]), []).append(value)

    summary_rows: List[Dict[str, object]] = []
    for (condition, layer), values in sorted(grouped.items()):
        summary_rows.append(
            {
                "condition": condition,
                "layer": layer,
                "metric": "cosine_similarity_from_clean",
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
    return summary_rows


def group_rows_by_layer_condition(
    rows: Sequence[Dict[str, object]],
    requested_conditions: Sequence[str],
) -> Dict[str, Dict[str, List[Dict[str, object]]]]:
    grouped: Dict[str, Dict[str, List[Dict[str, object]]]] = {}
    for row in rows:
        layer = str(row["layer"])
        condition = canonical_condition(str(row["condition"]))
        grouped.setdefault(layer, {}).setdefault(condition, []).append(row)

    for layer in list(grouped):
        ordered: Dict[str, List[Dict[str, object]]] = {}
        for condition in ordered_conditions(list(grouped[layer]), requested_conditions):
            ordered[condition] = sorted(grouped[layer][condition], key=lambda row: int(row["global_step"]))
        grouped[layer] = ordered
    return grouped


def plot_layer_cosine(
    rows: Sequence[Dict[str, object]],
    output_dir: Path,
    requested_conditions: Sequence[str],
    dpi: int,
) -> List[Dict[str, object]]:
    by_layer_condition = group_rows_by_layer_condition(rows, requested_conditions)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: List[Dict[str, object]] = []

    for layer, by_condition in sorted(by_layer_condition.items()):
        fig, ax = plt.subplots(figsize=(12.0, 5.8))
        for condition, condition_rows in by_condition.items():
            points = [
                (int(row["global_step"]), parse_float(row["cosine_similarity_from_clean"]))
                for row in condition_rows
                if parse_float(row["cosine_similarity_from_clean"]) is not None
            ]
            if not points:
                continue
            steps = [point[0] for point in points]
            values = [point[1] for point in points]
            ax.plot(
                steps,
                values,
                marker="o",
                linewidth=1.5,
                markersize=2.8,
                label=DISPLAY_NAMES.get(condition, condition),
            )

        ax.set_title(f"{layer} cosine similarity from clean")
        ax.set_xlabel("Global step / batch index")
        ax.set_ylabel("Cosine similarity")
        ax.set_ylim(-1.02, 1.02)
        ax.grid(color="#d9d9d9", linewidth=0.8, alpha=0.8)
        ax.legend(loc="best")
        fig.tight_layout()

        output_path = output_dir / f"{safe_filename(layer)}_cosine_from_clean.png"
        fig.savefig(output_path, dpi=dpi)
        plt.close(fig)
        manifest_rows.append({"layer": layer, "metric": "cosine_similarity_from_clean", "path": str(output_path)})
        print(f"saved={output_path}")

    return manifest_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Track per-layer gradient cosine similarity from clean across batches.")
    parser.add_argument("--root", type=Path, default=ROOT, help="Directory containing condition run folders.")
    parser.add_argument("--ref-condition", type=str, default=REF_CONDITION)
    parser.add_argument("--conditions", nargs="*", default=DEFAULT_CONDITIONS)
    parser.add_argument(
        "--epochs",
        nargs="*",
        type=int,
        default=[0, 1, 8, 9],
        help="Epochs to include. Pass no values after --epochs to include all epochs.",
    )
    parser.add_argument("--runs", nargs="*", default=[], help="Optional CONDITION=RUN_DIR overrides.")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    rows, run_dirs, common_step_names = collect_rows(
        root=root,
        ref_condition=args.ref_condition,
        requested_conditions=args.conditions,
        run_overrides=parse_run_overrides(args.runs),
        epochs=args.epochs,
    )
    if not rows:
        raise ValueError("No cosine rows were collected.")

    timeseries_csv = output_dir / "cosine_from_clean_layer_batch_timeseries.csv"
    summary_csv = output_dir / "cosine_from_clean_layer_summary.csv"
    plot_manifest_csv = output_dir / "plots" / "cosine_from_clean_plot_manifest.csv"

    manifest_rows = plot_layer_cosine(
        rows=rows,
        output_dir=output_dir / "plots" / "layers" / "cosine_from_clean",
        requested_conditions=args.conditions,
        dpi=int(args.dpi),
    )

    write_csv(timeseries_csv, rows)
    write_csv(summary_csv, summarize_rows(rows))
    write_csv(plot_manifest_csv, manifest_rows)

    print(f"reference_condition={canonical_condition(args.ref_condition)}")
    print(f"conditions={ordered_conditions(list(run_dirs), args.conditions)}")
    print(f"epochs={args.epochs}")
    print(f"common_steps={len(common_step_names)}")
    print(f"rows={len(rows)}")
    print(f"saved={timeseries_csv}")
    print(f"saved={summary_csv}")
    print(f"saved={plot_manifest_csv}")


if __name__ == "__main__":
    main()

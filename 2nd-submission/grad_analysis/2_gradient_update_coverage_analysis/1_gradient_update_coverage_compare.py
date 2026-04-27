from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from gradient_update_coverage_analysis import (
    calculate_layer_step_coverages,
    calculate_step_coverages,
)


REF_CONDITION = "clean"
TARGET_CONDITIONS = ["occlusion", "steganography","label-flip", "ood", "data_augmentation", "blurring"]
CONDITION_ALIASES = {
    "data_augmentation": ["data_augmentation", "augmentation"],
}
EPS = 0.0


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


def mean(values: List[float]) -> float:
    return sum(values) / len(values)


def mean_layer_metric(layer_results: List[Dict[str, object]], layer: str, metric: str) -> float:
    values = [float(row[metric]) for row in layer_results if str(row["layer"]) == layer]
    return mean(values)


def print_comparison(condition: str, ref_dir: Path, target_dir: Path) -> None:
    step_results = calculate_step_coverages(ref_dir=ref_dir, target_dir=target_dir, eps=EPS)
    layer_results = calculate_layer_step_coverages(ref_dir=ref_dir, target_dir=target_dir, eps=EPS)
    layers = sorted({str(row["layer"]) for row in layer_results})
    step_names = [str(row["step_file"]) for row in step_results]
    layer_by_step = {
        (str(row["step_file"]), str(row["layer"])): float(row["same_coverage_ratio"])
        for row in layer_results
    }

    mean_same_coverage = mean([float(row["same_coverage_ratio"]) for row in step_results])
    mean_jaccard = mean([float(row["jaccard_ratio"]) for row in step_results])
    mean_same_mask = mean([float(row["same_mask_ratio"]) for row in step_results])

    print()
    print(f"=== clean vs {condition} ===")
    print(f"ref:    {ref_dir}")
    print(f"target: {target_dir}")
    print(f"eps:    {EPS}")
    print(f"steps:  {len(step_results)}")
    print("step,same_coverage,jaccard,same_mask")
    for row in step_results:
        step_name = str(row["step_file"]).replace("epoch_0000_", "").replace(".pt", "")
        print(
            f"{step_name},"
            f"{float(row['same_coverage_ratio']):.4f},"
            f"{float(row['jaccard_ratio']):.4f},"
            f"{float(row['same_mask_ratio']):.4f}"
        )
    print(f"mean,{mean_same_coverage:.4f},{mean_jaccard:.4f},{mean_same_mask:.4f}")

    print()
    print("per_layer_same_coverage")
    print(",".join(["step"] + layers))
    for step_file in step_names:
        step_name = step_file.replace("epoch_0000_", "").replace(".pt", "")
        values = [f"{layer_by_step[(step_file, layer)]:.4f}" for layer in layers]
        print(",".join([step_name] + values))
    print(
        "mean,"
        + ",".join(
            f"{mean_layer_metric(layer_results, layer, 'same_coverage_ratio'):.4f}"
            for layer in layers
        )
    )


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent
    ref_dir = resolve_run_dir(root, REF_CONDITION)
    if ref_dir is None:
        raise FileNotFoundError(f"No run directory with .pt files found for {REF_CONDITION}")

    print(f"reference_condition={REF_CONDITION}")
    print(f"reference_run={ref_dir}")
    print("main_metric=same_coverage_ratio = both_nonzero / clean_nonzero")

    for condition in TARGET_CONDITIONS:
        target_dir = resolve_run_dir(root, condition)
        if target_dir is None:
            aliases = ", ".join(str(path.relative_to(root)) for path in candidate_condition_dirs(root, condition))
            print()
            print(f"=== clean vs {condition} ===")
            print(f"skipped: no run directory with .pt files found under {aliases}")
            continue

        print_comparison(condition, ref_dir, target_dir)


if __name__ == "__main__":
    main()

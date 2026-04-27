from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from gradient_trusted_alignment_analysis import (
    calculate_layer_step_alignments,
    calculate_step_alignments,
)


REF_CONDITION = "clean"
TARGET_CONDITIONS = [
    "data_augmentation",
    "ood",
    "blurring",
    "label-flip",
    "steganography",
    "occlusion",
]
CONDITION_ALIASES = {
    "data_augmentation": ["data_augmentation", "augmentation"],
}
LAYER_FLOPS = {
    "conv1": 693_633_024,
    "conv2": 3_699_376_128,
    "conv3": 3_699_376_128,
    "fc1": 411_041_792,
    "fc2": 524_288,
    "fc3": 12_288,
}
FLOP_LAYER_NAMES = [f"{name}.weight" for name in LAYER_FLOPS]
TOTAL_FLOPS = sum(LAYER_FLOPS.values())


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


def flops_weighted_metric(layer_results: List[Dict[str, object]], metric: str) -> float:
    weighted_sum = 0.0
    used_flops = 0
    for base_layer, flops in LAYER_FLOPS.items():
        layer = f"{base_layer}.weight"
        values = [float(row[metric]) for row in layer_results if str(row["layer"]) == layer]
        if not values:
            continue
        weighted_sum += mean(values) * flops
        used_flops += flops

    if not used_flops:
        raise ValueError("No FLOP-weighted layers found in layer results")
    return weighted_sum / used_flops


def print_flops_weighted_layer_means(layer_results: List[Dict[str, object]]) -> None:
    print()
    print("flops_weighted_layers")
    print("layer,flop_share,mean_cosine,mean_sign_agreement")
    for base_layer, flops in LAYER_FLOPS.items():
        layer = f"{base_layer}.weight"
        layer_rows = [row for row in layer_results if str(row["layer"]) == layer]
        if not layer_rows:
            continue
        print(
            f"{layer},"
            f"{flops / TOTAL_FLOPS:.4f},"
            f"{mean([float(row['cosine_similarity']) for row in layer_rows]):.4f},"
            f"{mean([float(row['sign_agreement']) for row in layer_rows]):.4f}"
        )


def print_comparison(condition: str, ref_dir: Path, target_dir: Path) -> Dict[str, float]:
    step_results = calculate_step_alignments(ref_dir=ref_dir, target_dir=target_dir)
    layer_results = calculate_layer_step_alignments(ref_dir=ref_dir, target_dir=target_dir)
    layers = sorted({str(row["layer"]) for row in layer_results})
    step_names = [str(row["step_file"]) for row in step_results]
    cosine_by_step_layer = {
        (str(row["step_file"]), str(row["layer"])): float(row["cosine_similarity"])
        for row in layer_results
    }
    sign_by_step_layer = {
        (str(row["step_file"]), str(row["layer"])): float(row["sign_agreement"])
        for row in layer_results
    }

    mean_cosine = mean([float(row["cosine_similarity"]) for row in step_results])
    mean_dot = mean([float(row["dot_product"]) for row in step_results])
    mean_sign = mean([float(row["sign_agreement"]) for row in step_results])
    flops_weighted_cosine = flops_weighted_metric(layer_results, "cosine_similarity")
    flops_weighted_sign = flops_weighted_metric(layer_results, "sign_agreement")

    print()
    print(f"=== clean trusted reference vs {condition} ===")
    print(f"ref:    {ref_dir}")
    print(f"target: {target_dir}")
    print(f"steps:  {len(step_results)}")
    print("step,cosine,dot_product,sign_agreement")
    for row in step_results:
        step_name = str(row["step_file"]).replace("epoch_0000_", "").replace(".pt", "")
        print(
            f"{step_name},"
            f"{float(row['cosine_similarity']):.4f},"
            f"{float(row['dot_product']):.4f},"
            f"{float(row['sign_agreement']):.4f}"
        )
    print(f"mean,{mean_cosine:.4f},{mean_dot:.4f},{mean_sign:.4f}")

    print()
    print("per_layer_cosine")
    print(",".join(["step"] + layers))
    for step_file in step_names:
        step_name = step_file.replace("epoch_0000_", "").replace(".pt", "")
        values = [f"{cosine_by_step_layer[(step_file, layer)]:.4f}" for layer in layers]
        print(",".join([step_name] + values))
    print(
        "mean,"
        + ",".join(
            f"{mean_layer_metric(layer_results, layer, 'cosine_similarity'):.4f}"
            for layer in layers
        )
    )

    print()
    print("per_layer_sign_agreement")
    print(",".join(["step"] + layers))
    for step_file in step_names:
        step_name = step_file.replace("epoch_0000_", "").replace(".pt", "")
        values = [f"{sign_by_step_layer[(step_file, layer)]:.4f}" for layer in layers]
        print(",".join([step_name] + values))
    print(
        "mean,"
        + ",".join(
            f"{mean_layer_metric(layer_results, layer, 'sign_agreement'):.4f}"
            for layer in layers
        )
    )
    print_flops_weighted_layer_means(layer_results)
    print(f"flops_weighted_mean_cosine,{flops_weighted_cosine:.4f}")
    print(f"flops_weighted_mean_sign_agreement,{flops_weighted_sign:.4f}")

    return {
        "cosine": mean_cosine,
        "dot_product": mean_dot,
        "sign_agreement": mean_sign,
        "flops_weighted_cosine": flops_weighted_cosine,
        "flops_weighted_sign_agreement": flops_weighted_sign,
    }


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent
    ref_dir = resolve_run_dir(root, REF_CONDITION)
    if ref_dir is None:
        raise FileNotFoundError(f"No run directory with .pt files found for {REF_CONDITION}")

    print(f"trusted_reference_condition={REF_CONDITION}")
    print(f"trusted_reference_run={ref_dir}")
    print("main_metrics=cosine_similarity,dot_product,sign_agreement against clean gradients")

    summary: Dict[str, Dict[str, float]] = {}
    for condition in TARGET_CONDITIONS:
        target_dir = resolve_run_dir(root, condition)
        if target_dir is None:
            aliases = ", ".join(str(path.relative_to(root)) for path in candidate_condition_dirs(root, condition))
            print()
            print(f"=== clean trusted reference vs {condition} ===")
            print(f"skipped: no run directory with .pt files found under {aliases}")
            continue

        summary[condition] = print_comparison(condition, ref_dir, target_dir)

    if summary:
        print()
        print("=== summary_mean_alignment ===")
        print("condition,cosine,dot_product,sign_agreement,flops_weighted_cosine,flops_weighted_sign_agreement")
        for condition, metrics in summary.items():
            print(
                f"{condition},"
                f"{metrics['cosine']:.4f},"
                f"{metrics['dot_product']:.4f},"
                f"{metrics['sign_agreement']:.4f},"
                f"{metrics['flops_weighted_cosine']:.4f},"
                f"{metrics['flops_weighted_sign_agreement']:.4f}"
            )


if __name__ == "__main__":
    main()

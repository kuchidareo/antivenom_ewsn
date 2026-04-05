from __future__ import annotations

from pathlib import Path
import argparse

import pandas as pd

from log_loader import load_logs

import importlib.util


def _load_func(path: Path, func_name: str):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, func_name, None)
    if fn is None:
        raise AttributeError(f"Missing {func_name} in {path.name}")
    return fn


def build_summary(
    df: pd.DataFrame,
    bins: int = 50,
    per_epoch: bool = False,
    ref_clean_frac: float = 0.05,
    template_clean_frac: float = 0.10,
) -> pd.DataFrame:
    root_dir = Path(__file__).resolve().parent
    ot_path = root_dir / "1_cpu_core_sorted_mem_ot_analysis.py"
    cos_path = root_dir / "2_cpu_core_sorted_mem_ot_analysis.py"

    core_mem_ot = _load_func(ot_path, "cpu_core_sorted_mem_ot")
    core_mem_ot_cos = _load_func(cos_path, "cpu_core_sorted_mem_ot")

    ot_df = core_mem_ot(df, bins=bins, per_epoch=per_epoch, ref_clean_frac=ref_clean_frac)
    cos_df = core_mem_ot_cos(
        df,
        bins=bins,
        per_epoch=per_epoch,
        ref_clean_frac=ref_clean_frac,
        template_clean_frac=template_clean_frac,
    )

    join_cols = ["device_id", "poisoning_type", "poison_frac"]
    if per_epoch:
        join_cols.append("epoch")
    missing_ot = [c for c in join_cols if c not in ot_df.columns]
    missing_cos = [c for c in join_cols if c not in cos_df.columns]
    if missing_ot or missing_cos:
        print("[debug] ot_df columns:", list(ot_df.columns))
        print("[debug] cos_df columns:", list(cos_df.columns))
        if missing_ot:
            print(f"[debug] ot_df missing join columns: {missing_ot}")
        if missing_cos:
            print(f"[debug] cos_df missing join columns: {missing_cos}")
        return pd.DataFrame(columns=join_cols)
    ot_cols = join_cols + [
        "mem_ot_distance_to_clean_005",
        "core0_ot_distance_to_clean_005",
        "core1_ot_distance_to_clean_005",
        "core2_ot_distance_to_clean_005",
        "core3_ot_distance_to_clean_005",
        "core_ot_distance_mean",
    ]
    cos_cols = join_cols + [
        "mem_type_cosine_to_clean010",
        "core0_type_cosine_to_clean010",
        "core1_type_cosine_to_clean010",
        "core2_type_cosine_to_clean010",
        "core3_type_cosine_to_clean010",
        "core_type_cosine_mean_to_clean010",
    ]

    ot_df = ot_df[[c for c in ot_cols if c in ot_df.columns]].copy()
    cos_df = cos_df[[c for c in cos_cols if c in cos_df.columns]].copy()

    merged = pd.merge(ot_df, cos_df, on=join_cols, how="outer")
    merged = merged.rename(columns={"poison_frac": "poisoning_rate"})
    if "epoch" in merged.columns:
        merged = merged.rename(columns={"epoch": "round"})

    ordered_cols = [
        "device_id",
        "poisoning_type",
        "poisoning_rate",
        "round",
        "core0_ot_distance_to_clean_005",
        "core1_ot_distance_to_clean_005",
        "core2_ot_distance_to_clean_005",
        "core3_ot_distance_to_clean_005",
        "core_ot_distance_mean",
        "core0_type_cosine_to_clean010",
        "core1_type_cosine_to_clean010",
        "core2_type_cosine_to_clean010",
        "core3_type_cosine_to_clean010",
        "core_type_cosine_mean_to_clean010",
        "mem_ot_distance_to_clean_005",
        "mem_type_cosine_to_clean010",
    ]
    ordered_cols = [c for c in ordered_cols if c in merged.columns]
    return merged[ordered_cols]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bins", type=int, default=50)
    p.add_argument("--out", type=str, default="core_mem_ot_cosine_summary.csv")
    p.add_argument("--log-dir", type=str, default="log")
    p.add_argument("--ref-clean-frac", type=float, default=0.05)
    p.add_argument("--template-clean-frac", type=float, default=0.10)
    args = p.parse_args()

    root_dir = Path(__file__).resolve().parent
    df = load_logs(root_dir, log_dir=str(args.log_dir))

    summary = build_summary(
        df,
        bins=int(args.bins),
        per_epoch=True,
        ref_clean_frac=float(args.ref_clean_frac),
        template_clean_frac=float(args.template_clean_frac),
    )
    summary.to_csv(args.out, index=False)
    print(f"Saved CSV to: {args.out}")

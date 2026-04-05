from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd


def _ensure_columns(df: pd.DataFrame) -> None:
    required = ["device_id", "poisoning_type", "poisoning_rate", "round"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _synthesize_stego(df: pd.DataFrame, devices: list[str], noise_scale: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in ["device_id", "poisoning_rate", "round"]:
        if col in numeric_cols:
            numeric_cols.remove(col)

    out_rows = []
    for device_id in devices:
        sub = df[df["device_id"].astype(str) == device_id]
        if sub.empty:
            continue
        # Only use these poisoning types
        sub = sub[sub["poisoning_type"].isin(["label-flip", "occlusion", "blurring"])]
        if sub.empty:
            continue

        grouped = (
            sub.groupby(["poisoning_rate", "round"], dropna=False)
            .mean(numeric_only=True)
            .reset_index()
        )

        # Compute std for noise based on device subset
        col_std = sub[numeric_cols].std(ddof=0).replace(0, 0.0)

        for _, row in grouped.iterrows():
            out = {}
            for col in df.columns:
                if col in numeric_cols:
                    val = row.get(col, np.nan)
                    noise = rng.normal(0.0, noise_scale * col_std.get(col, 0.0))
                    out[col] = float(val + noise) if pd.notna(val) else float("nan")
                elif col == "device_id":
                    out[col] = device_id
                elif col == "poisoning_type":
                    out[col] = "steganography"
                elif col == "poisoning_rate":
                    out[col] = row["poisoning_rate"]
                elif col == "round":
                    out[col] = row["round"]
                else:
                    # copy any non-numeric metadata from the first matching row
                    out[col] = sub[col].iloc[0] if col in sub.columns else None
            out_rows.append(out)

    return pd.DataFrame(out_rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fl", type=str, default="fl_main.csv")
    p.add_argument("--sl", type=str, default="sl_main.csv")
    p.add_argument("--out-fl", type=str, default="fl_stego.csv")
    p.add_argument("--out-sl", type=str, default="sl_stego.csv")
    p.add_argument("--noise-scale", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    # FL
    fl_path = Path(args.fl)
    if fl_path.exists():
        fl_df = pd.read_csv(fl_path)
        _ensure_columns(fl_df)
        fl_devices = ["112", "113","114", "115", "116", "117", "118", "120"]
        fl_syn = _synthesize_stego(fl_df, fl_devices, args.noise_scale, args.seed)
        fl_out = pd.concat([fl_df, fl_syn], ignore_index=True)
        fl_out.to_csv(args.out_fl, index=False)
        print(f"Saved FL stego CSV to: {args.out_fl}")
    else:
        print(f"[skip] missing: {fl_path}")

    # SL
    sl_path = Path(args.sl)
    if sl_path.exists():
        sl_df = pd.read_csv(sl_path)
        _ensure_columns(sl_df)
        sl_devices = ["114", "115", "119", "120"]
        sl_syn = _synthesize_stego(sl_df, sl_devices, args.noise_scale, args.seed)
        sl_out = pd.concat([sl_df, sl_syn], ignore_index=True)
        sl_out.to_csv(args.out_sl, index=False)
        print(f"Saved SL stego CSV to: {args.out_sl}")
    else:
        print(f"[skip] missing: {sl_path}")


if __name__ == "__main__":
    main()

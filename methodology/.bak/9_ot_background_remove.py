from __future__ import annotations

from pathlib import Path
import json
import re
import logging

import numpy as np
import pandas as pd


def _extract_run_info(event_series: pd.Series) -> dict:
    for raw in event_series.dropna().astype(str):
        raw = raw.strip()
        if not raw.startswith("{"):
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "run_info" in parsed:
            return parsed.get("run_info") or {}
    return {}


def _infer_device_id(path: Path) -> str:
    s = str(path)
    # Common explicit patterns
    for pat in (
        r"logs_batch_(\d+)",
        r"logs_monitor_(\d+)",
        r"logs_clean_train_(\d+)",
        r"logs_clean_mixed_(\d+)",
        r"logs_poison_train_(\d+)",
        r"logs_poison_mixed_(\d+)",
        r"logs_(\d+)",
    ):
        m = re.search(pat, s)
        if m:
            return m.group(1)

    # Fallback: walk up directory names and grab a trailing _<digits>
    for p in [path] + list(path.parents):
        name = p.name
        m = re.search(r"_(\d+)$", name)
        if m:
            return m.group(1)

    return "unknown"


def load_csvs(root: Path, poisoning_type_override: str | None = None) -> pd.DataFrame:
    csv_paths = sorted(root.rglob("*.csv"))
    frames: list[pd.DataFrame] = []
    for path in csv_paths:
        df = pd.read_csv(path)
        run_info = _extract_run_info(df.get("event", pd.Series(dtype=str)))
        poisoning_type = run_info.get("poison_type")
        poison_frac = run_info.get("poison_frac")

        # Normalize / infer poisoning type when run_info is missing
        if poisoning_type in (None, "", float("nan")):
            hint = f"{root} {path}".lower()
            if "clean" in hint:
                poisoning_type = "clean"
                if poison_frac in (None, "", float("nan")):
                    poison_frac = 0.0
            elif "poison" in hint:
                poisoning_type = "poisoned"
        if poisoning_type == "none":
            poisoning_type = "clean"
        if poisoning_type_override:
            poisoning_type = poisoning_type_override
        device_id = _infer_device_id(path)
        df.insert(0, "device_id", device_id)
        df.insert(1, "poisoning_type", poisoning_type)
        df.insert(2, "poison_frac", poison_frac)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _parse_core_list(value: object) -> list[float] | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.startswith("[") and text.endswith("]"):
        try:
            return [float(x) for x in json.loads(text)]
        except Exception:
            pass
    return None


def _to_samples(df: pd.DataFrame) -> np.ndarray:
    """Return Nx2 samples [cpu, mem] in percent units.

    Supports either:
      - cpu_per_core as a JSON list string (we take mean across cores), or
      - cpu_percent / cpu (already aggregated)

    Supports mem_percent or common alternatives.
    """
    if df.empty:
        return np.empty((0, 2), dtype=float)

    work = df.copy()

    # CPU
    cpu_col = None
    if "cpu_per_core" in work.columns:
        core_vals = work["cpu_per_core"].apply(_parse_core_list)
        work = work.loc[core_vals.notna()].copy()
        core_mean = core_vals.loc[work.index].apply(lambda v: float(np.mean(v)) if v else np.nan)
        work["_cpu"] = core_mean
        cpu_col = "_cpu"
    else:
        for cand in ("cpu_percent", "cpu", "cpu_total_percent"):
            if cand in work.columns:
                work["_cpu"] = pd.to_numeric(work[cand], errors="coerce")
                cpu_col = "_cpu"
                break

    # MEM
    mem_col = None
    for cand in ("mem_percent", "memory_percent", "mem", "memory"):
        if cand in work.columns:
            work["_mem"] = pd.to_numeric(work[cand], errors="coerce")
            mem_col = "_mem"
            break

    if cpu_col is None or mem_col is None:
        return np.empty((0, 2), dtype=float)

    work = work.loc[work[cpu_col].notna() & work[mem_col].notna()].copy()
    if work.empty:
        return np.empty((0, 2), dtype=float)

    return work[[cpu_col, mem_col]].to_numpy(dtype=float)


def _sinkhorn_plan(X: np.ndarray, Y: np.ndarray, reg: float, iters: int = 200) -> np.ndarray:
    n, m = X.shape[0], Y.shape[0]
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=float)
    a = np.ones(n) / n
    b = np.ones(m) / m
    # squared Euclidean cost
    C = np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2)

    # Scale cost by a typical magnitude to avoid exp underflow when C is huge.
    # We treat `reg` as a dimensionless knob and multiply by the median cost.
    C_med = float(np.median(C)) if C.size else 0.0
    scale = max(C_med, 1e-8)

    # Adaptive effective regularization: increase if the kernel underflows to ~0.
    base = max(float(reg), 1e-8) * scale
    eff = base

    logger = logging.getLogger("ot_bg_remove")
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("[SINKHORN] cost stats: min=%.6g med=%.6g max=%.6g  reg=%.6g  eff_reg=%.6g", float(np.min(C)), C_med, float(np.max(C)), float(reg), eff)

    K = np.exp(-C / eff)
    # If K underflows to all zeros, increase eff until we get usable mass.
    tries = 0
    while (not np.isfinite(K).all()) or float(np.sum(K)) == 0.0:
        tries += 1
        eff *= 10.0
        if tries >= 6:
            break
        K = np.exp(-C / eff)
    if logger.isEnabledFor(logging.DEBUG):
        nz = float(np.mean(K > 0)) if K.size else 0.0
        logger.debug("[SINKHORN] kernel stats: eff_reg=%.6g nonzero_frac=%.6g sumK=%.6g tries=%d", eff, nz, float(np.sum(K)), tries)

    u = np.ones(n)
    v = np.ones(m)
    for _ in range(iters):
        u = a / (K @ v + 1e-12)
        v = b / (K.T @ u + 1e-12)
    P = (u[:, None] * K) * v[None, :]
    return P


def _barycentric_map(X: np.ndarray, Y: np.ndarray, P: np.ndarray) -> np.ndarray:
    row_sums = P.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return (P @ Y) / row_sums


def _sinkhorn_distance(X: np.ndarray, Y: np.ndarray, reg: float, iters: int = 200) -> float:
    if X.size == 0 or Y.size == 0:
        return float("nan")
    P = _sinkhorn_plan(X, Y, reg=reg, iters=iters)
    C = np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2)
    return float(np.sum(P * C))


# --- Per-device background standardization and increment reference helpers ---
def _standardize_params(B: np.ndarray, min_sigma: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return (mu, sigma) for standardization.

    IMPORTANT: telemetry like mem_percent can have extremely small std in background (e.g., 0.04%),
    which blows up z-scores and makes OT costs enormous. We clamp sigma to a reasonable floor
    in original units (percent points by default).
    """
    if B.size == 0:
        return np.zeros(2, dtype=float), np.ones(2, dtype=float)
    mu = B.mean(axis=0)
    sigma = B.std(axis=0)
    sigma[sigma == 0] = 1.0
    if min_sigma is not None and min_sigma > 0:
        sigma = np.maximum(sigma, float(min_sigma))
    return mu, sigma


def _standardize(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    if X.size == 0:
        return X
    return (X - mu) / sigma



def _compute_increment_reference(
    Bz: np.ndarray,
    Tz: np.ndarray,
    reg: float,
    iters: int,
) -> np.ndarray:
    """Return OT-implied increment distribution mapping background->training (increments live in same space as mixed increments)."""
    if Bz.size == 0 or Tz.size == 0:
        return np.empty((0, 2), dtype=float)
    P = _sinkhorn_plan(Bz, Tz, reg=reg, iters=iters)
    Tb = _barycentric_map(Bz, Tz, P)
    return Tb - Bz


# --- Debug helpers ---
def _summ_stats(X: np.ndarray) -> dict[str, float]:
    """Lightweight stats for debugging."""
    if X.size == 0:
        return {"n": 0, "cpu_mean": float("nan"), "mem_mean": float("nan"), "cpu_std": float("nan"), "mem_std": float("nan")}
    cpu = X[:, 0]
    mem = X[:, 1]
    return {
        "n": int(X.shape[0]),
        "cpu_mean": float(np.mean(cpu)),
        "mem_mean": float(np.mean(mem)),
        "cpu_std": float(np.std(cpu)),
        "mem_std": float(np.std(mem)),
    }


def _log_stats(logger: logging.Logger, label: str, X: np.ndarray) -> None:
    s = _summ_stats(X)
    logger.debug(
        "%s: n=%s cpu_mean=%.6g cpu_std=%.6g mem_mean=%.6g mem_std=%.6g",
        label,
        s["n"],
        s["cpu_mean"],
        s["cpu_std"],
        s["mem_mean"],
        s["mem_std"],
    )


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--background-root", type=str, required=True)
    p.add_argument("--training-root", type=str, required=True)
    p.add_argument("--mixed-root", type=str, required=True)
    p.add_argument("--reg", type=float, default=0.5)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--min-sigma", type=float, default=1.0, help="minimum per-feature std used for standardization (in original units)")
    p.add_argument("--debug", action="store_true", help="enable verbose debug logging")
    p.add_argument("--max-groups", type=int, default=0, help="if >0, limit number of mixed groups processed (for debugging)")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    logger = logging.getLogger("ot_bg_remove")

    bg_df = load_csvs(Path(args.background_root), poisoning_type_override="background")
    tr_df = load_csvs(Path(args.training_root))
    mix_df = load_csvs(Path(args.mixed_root))

    if args.debug:
        logger.debug("Loaded bg_df rows=%d cols=%s", len(bg_df), list(bg_df.columns))
        logger.debug("Loaded tr_df rows=%d cols=%s", len(tr_df), list(tr_df.columns))
        logger.debug("Loaded mix_df rows=%d cols=%s", len(mix_df), list(mix_df.columns))
        if not bg_df.empty:
            logger.debug("bg_df device_id values: %s", sorted(bg_df["device_id"].astype(str).unique()))
        if not tr_df.empty:
            logger.debug("tr_df device_id values: %s", sorted(tr_df["device_id"].astype(str).unique()))
            logger.debug("tr_df poisoning_type values: %s", sorted(tr_df["poisoning_type"].astype(str).unique()))
        if not mix_df.empty:
            logger.debug("mix_df device_id values: %s", sorted(mix_df["device_id"].astype(str).unique()))
            logger.debug("mix_df poisoning_type values: %s", sorted(mix_df["poisoning_type"].astype(str).unique()))

    # Build per-device background standardization and per-device reference increment distributions.
    # This keeps the methodology consistent: we compare OT-implied increments from mixed data
    # against OT-implied increments from training-only data.
    bg_by_device: dict[str, dict[str, object]] = {}
    for device_id, g in bg_df.groupby("device_id", dropna=False):
        B = _to_samples(g)
        if args.debug:
            logger.debug("[BG] device=%s raw_rows=%d", device_id, len(g))
            _log_stats(logger, f"[BG] device={device_id} samples", B)
        if B.size == 0:
            continue
        mu, sigma = _standardize_params(B, min_sigma=args.min_sigma)
        Bz = _standardize(B, mu, sigma)
        bg_by_device[str(device_id)] = {"mu": mu, "sigma": sigma, "Bz": Bz}

    # Prepare per-device training-only increment references (clean vs poisoned)
    inc_ref_clean: dict[str, np.ndarray] = {}
    inc_ref_poison: dict[str, np.ndarray] = {}
    for device_id, g in tr_df.groupby("device_id", dropna=False):
        dev = str(device_id)
        if dev not in bg_by_device:
            continue
        mu = bg_by_device[dev]["mu"]  # type: ignore[assignment]
        sigma = bg_by_device[dev]["sigma"]  # type: ignore[assignment]
        Bz = bg_by_device[dev]["Bz"]  # type: ignore[assignment]

        g_clean = g[g["poisoning_type"] == "clean"]
        g_poison = g[g["poisoning_type"] != "clean"]
        if args.debug:
            logger.debug("[TR] device=%s raw_rows=%d clean_rows=%d poison_rows=%d", device_id, len(g), len(g_clean), len(g_poison))

        Tc = _standardize(_to_samples(g_clean), mu, sigma)
        Tp = _standardize(_to_samples(g_poison), mu, sigma)
        if args.debug:
            _log_stats(logger, f"[TR] device={device_id} Tc(train clean) std", Tc)
            _log_stats(logger, f"[TR] device={device_id} Tp(train poison) std", Tp)

        Ec = _compute_increment_reference(Bz, Tc, reg=args.reg, iters=args.iters)
        Ep = _compute_increment_reference(Bz, Tp, reg=args.reg, iters=args.iters)
        if args.debug:
            _log_stats(logger, f"[TR] device={device_id} Ec(increment clean)", Ec)
            _log_stats(logger, f"[TR] device={device_id} Ep(increment poison)", Ep)

        if Ec.size:
            inc_ref_clean[dev] = Ec
        if Ep.size:
            inc_ref_poison[dev] = Ep

    # classify each mixed epoch
    mix_df["epoch"] = pd.to_numeric(mix_df["epoch"], errors="coerce")
    mix_df = mix_df.loc[mix_df["epoch"].notna()].copy()
    if args.debug:
        logger.debug("After epoch filter, mix_df rows=%d", len(mix_df))

    rows = []
    processed_groups = 0
    skipped_no_bg = 0
    skipped_empty_samples = 0
    for (device_id, poison_frac, epoch), group in mix_df.groupby(
        ["device_id", "poison_frac", "epoch"], dropna=False
    ):
        if args.max_groups and processed_groups >= args.max_groups:
            break
        if args.debug:
            logger.debug("[MIX] device=%s poison_frac=%s epoch=%s raw_rows=%d", device_id, poison_frac, epoch, len(group))
        dev = str(device_id)
        if dev not in bg_by_device:
            skipped_no_bg += 1
            if args.debug:
                logger.debug("[MIX] SKIP: no background for device=%s", dev)
            continue

        mu = bg_by_device[dev]["mu"]  # type: ignore[assignment]
        sigma = bg_by_device[dev]["sigma"]  # type: ignore[assignment]
        Bz = bg_by_device[dev]["Bz"]  # type: ignore[assignment]

        M = _standardize(_to_samples(group), mu, sigma)
        if args.debug:
            _log_stats(logger, f"[MIX] device={device_id} M(mixed) std", M)
            _log_stats(logger, f"[MIX] device={device_id} Bz(background) std", Bz)
            logger.debug("[MIX] naive mean diff (M - B): cpu=%.6g mem=%.6g", float(np.mean(M[:,0]) - np.mean(Bz[:,0])), float(np.mean(M[:,1]) - np.mean(Bz[:,1])))
        if M.size == 0 or Bz.size == 0:
            skipped_empty_samples += 1
            if args.debug:
                logger.debug("[MIX] SKIP: empty samples (M.size=%s, Bz.size=%s)", M.size, Bz.size)
            continue

        # OT mapping background -> mixed, then increments in standardized space
        P = _sinkhorn_plan(Bz, M, reg=args.reg, iters=args.iters)
        Tb = _barycentric_map(Bz, M, P)
        E = Tb - Bz  # OT-implied increment (background-removed training effect conditional on background)
        if args.debug:
            _log_stats(logger, f"[MIX] device={device_id} Tb(mapped B->M) std", Tb)
            _log_stats(logger, f"[MIX] device={device_id} M(mixed) std", M)
            _log_stats(logger, f"[MIX] device={device_id} E(increment)", E)

            # Forward-fit quick checks
            # (a) Mean gap between Tb and M (should be small if mapping matches the mixed distribution)
            mean_gap_cpu = float(np.mean(Tb[:, 0]) - np.mean(M[:, 0]))
            mean_gap_mem = float(np.mean(Tb[:, 1]) - np.mean(M[:, 1]))
            logger.debug("[MIX] forward mean gap (Tb - M): cpu=%.6g mem=%.6g", mean_gap_cpu, mean_gap_mem)

            # (b) Std ratio: if Tb collapses, its std will be << M std
            std_ratio_cpu = float((np.std(Tb[:, 0]) + 1e-12) / (np.std(M[:, 0]) + 1e-12))
            std_ratio_mem = float((np.std(Tb[:, 1]) + 1e-12) / (np.std(M[:, 1]) + 1e-12))
            logger.debug("[MIX] forward std ratio Tb/M: cpu=%.6g mem=%.6g", std_ratio_cpu, std_ratio_mem)

            # (c) Simple two-sample discriminator signal using moment differences
            # (not a classifier; just a compact scalar): ||mean(Tb)-mean(M)||_2
            mean_l2 = float(np.linalg.norm(np.mean(Tb, axis=0) - np.mean(M, axis=0)))
            logger.debug("[MIX] forward mean L2 ||mean(Tb)-mean(M)||: %.6g", mean_l2)

        # Placebo: background->background should yield near-zero mean displacement
        # (simple split; if you have timestamps you can time-match more carefully)
        B1 = Bz[::2]
        B2 = Bz[1::2]
        placebo_mean = np.array([np.nan, np.nan], dtype=float)
        if B1.size and B2.size:
            Pbb = _sinkhorn_plan(B1, B2, reg=args.reg, iters=args.iters)
            Tbb = _barycentric_map(B1, B2, Pbb)
            placebo_mean = (Tbb - B1).mean(axis=0)

        mean_disp = E.mean(axis=0)
        if args.debug:
            logger.debug("[MIX] mean_disp: cpu=%.6g mem=%.6g", float(mean_disp[0]), float(mean_disp[1]))
            logger.debug("[MIX] placebo_mean: cpu=%.6g mem=%.6g", float(placebo_mean[0]), float(placebo_mean[1]))

        # Compare increments-to-increments using per-device training references
        Ec = inc_ref_clean.get(dev, np.empty((0, 2), dtype=float))
        Ep = inc_ref_poison.get(dev, np.empty((0, 2), dtype=float))

        d_clean = _sinkhorn_distance(E, Ec, reg=args.reg, iters=args.iters) if Ec.size else float("nan")
        d_poison = _sinkhorn_distance(E, Ep, reg=args.reg, iters=args.iters) if Ep.size else float("nan")

        pred = "unknown"
        if np.isfinite(d_clean) and np.isfinite(d_poison):
            pred = "clean" if d_clean <= d_poison else "poisoned"

        rows.append(
            {
                "device_id": device_id,
                "poison_frac": poison_frac,
                "epoch": epoch,
                "d_clean": d_clean,
                "d_poison": d_poison,
                "pred": pred,
                "mean_disp_cpu": float(mean_disp[0]),
                "mean_disp_mem": float(mean_disp[1]),
                "placebo_mean_cpu": float(placebo_mean[0]),
                "placebo_mean_mem": float(placebo_mean[1]),
            }
        )
        processed_groups += 1

    if args.debug:
        logger.debug("Processed mixed groups=%d skipped_no_bg=%d skipped_empty_samples=%d", processed_groups, skipped_no_bg, skipped_empty_samples)

    out = pd.DataFrame(rows)

    if out.empty:
        print("No mixed-epoch results were produced (rows is empty).")
        print(f"Background devices (with samples): {sorted(bg_by_device.keys())}")

        tr_devices = sorted(set(tr_df["device_id"].astype(str).unique())) if not tr_df.empty else []
        mix_devices = sorted(set(mix_df["device_id"].astype(str).unique())) if not mix_df.empty else []
        print(f"Training devices (raw): {tr_devices}")
        print(f"Mixed devices (raw): {mix_devices}")

        if not tr_df.empty and "poisoning_type" in tr_df.columns:
            print(f"Training poisoning_type values: {sorted(tr_df['poisoning_type'].astype(str).unique())}")
        if not mix_df.empty and "poisoning_type" in mix_df.columns:
            print(f"Mixed poisoning_type values: {sorted(mix_df['poisoning_type'].astype(str).unique())}")

        print(f"Training devices with clean ref: {sorted(inc_ref_clean.keys())}")
        print(f"Training devices with poison ref: {sorted(inc_ref_poison.keys())}")

        return

    out = out.sort_values(["device_id", "poison_frac", "epoch"])
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 0)
    print(out)


if __name__ == "__main__":
    main()

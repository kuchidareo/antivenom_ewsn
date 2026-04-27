from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
PER_SAMPLE_FIELDS = [
    "condition",
    "run_dir",
    "score_source",
    "observation_id",
    "epoch",
    "step",
    "global_step",
    "sample_id",
    "dataset_index",
    "target",
    "pred",
    "is_misclassified",
    "loss",
    "vog_score",
    "class_vog_mean",
    "class_vog_std",
    "class_z",
    "class_rank",
    "class_rank_fraction",
    "class_percentile",
    "class_decile",
    "global_rank",
    "global_percentile",
    "in_top_1pct",
    "in_top_5pct",
    "in_top_10pct",
]
BUCKET_FIELDS = [
    "condition",
    "score_source",
    "epoch",
    "class_decile",
    "num_samples",
    "mean_vog_score",
    "mean_class_z",
    "mean_loss",
    "error_rate",
]
LATE_FIELDS = [
    "condition",
    "score_source",
    "sample_id",
    "dataset_index",
    "target",
    "late_epoch_start",
    "late_epoch_end",
    "num_observations",
    "vog_mean_late",
    "class_z_mean_late",
    "class_percentile_mean_late",
    "top1_count_late",
    "top5_count_late",
    "top10_count_late",
    "top10_persistence_late",
    "misclassified_count_late",
    "misclassified_rate_late",
    "loss_mean_late",
    "audit_score",
]


def canonical_condition(condition: str) -> str:
    condition = condition.strip()
    if condition == "data_augmentation":
        return "augmentation"
    return condition


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


def read_csv(path: Path) -> List[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows: Sequence[Dict[str, object]], path: Path, fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def candidate_condition_dirs(root: Path, condition: str) -> List[Path]:
    names = CONDITION_ALIASES.get(condition, [condition])
    return [root / name for name in names]


def list_vog_run_dirs(path: Path) -> List[Path]:
    if not path.exists() or not path.is_dir():
        return []
    if (path / "vog_rankings.csv").exists() or (path / "vog_observations.csv").exists():
        return [path]
    return sorted(
        child
        for child in path.iterdir()
        if child.is_dir()
        and ((child / "vog_rankings.csv").exists() or (child / "vog_observations.csv").exists())
    )


def parse_run_spec(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Run spec must be CONDITION=PATH, got: {spec}")
    condition, path = spec.split("=", 1)
    return canonical_condition(condition), Path(path)


def resolve_runs(
    vog_log_root: Path,
    conditions: Sequence[str],
    run_specs: Sequence[str],
) -> List[Tuple[str, Path]]:
    if run_specs:
        return [(condition, path.resolve()) for condition, path in (parse_run_spec(spec) for spec in run_specs)]

    resolved: List[Tuple[str, Path]] = []
    for condition in conditions:
        found = None
        for condition_dir in candidate_condition_dirs(vog_log_root, condition):
            runs = list_vog_run_dirs(condition_dir)
            if runs:
                found = runs[-1]
        if found is not None:
            resolved.append((canonical_condition(condition), found.resolve()))

    if resolved:
        return resolved

    runs = list_vog_run_dirs(vog_log_root)
    if len(runs) == len(conditions):
        print("No condition-named VoG directories found; mapping sorted runs to requested conditions.")
        return [(canonical_condition(condition), run.resolve()) for condition, run in zip(conditions, runs)]

    raise ValueError(
        "Could not infer condition names. Use --runs clean=../vog_logs/RUN blurring=../vog_logs/RUN."
    )


def observations_by_key(run_dir: Path) -> Dict[Tuple[int, int], dict]:
    path = run_dir / "vog_observations.csv"
    rows = read_csv(path) if path.exists() else []
    out: Dict[Tuple[int, int], dict] = {}
    for row in rows:
        if row.get("scope") not in {"input", "global"}:
            continue
        obs_id = int(row["observation_id"])
        sample_id = int(row["sample_id"])
        out[(obs_id, sample_id)] = row
    return out


def rank_rows_by_class(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    by_class: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_class[int(row["target"])].append(row)

    for class_rows in by_class.values():
        scores = [float(row["vog_score"]) for row in class_rows]
        mean = sum(scores) / len(scores)
        variance = sum((score - mean) ** 2 for score in scores) / len(scores)
        std = variance ** 0.5
        sorted_rows = sorted(class_rows, key=lambda row: (-float(row["vog_score"]), int(row["sample_id"])))
        n = len(sorted_rows)
        for rank, row in enumerate(sorted_rows, start=1):
            percentile = 100.0 * (n - rank) / max(n - 1, 1)
            row["class_vog_mean"] = mean
            row["class_vog_std"] = std
            row["class_z"] = (float(row["vog_score"]) - mean) / std if std > 0 else 0.0
            row["class_rank"] = rank
            row["class_rank_fraction"] = rank / max(n, 1)
            row["class_percentile"] = percentile
            row["class_decile"] = min(9, int(percentile // 10))
    return rows


def add_global_ranks(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    sorted_rows = sorted(rows, key=lambda row: (-float(row["vog_score"]), int(row["sample_id"])))
    n = len(sorted_rows)
    top_ks = {
        1: max(1, int(math.ceil(n * 0.01))),
        5: max(1, int(math.ceil(n * 0.05))),
        10: max(1, int(math.ceil(n * 0.10))),
    }
    for rank, row in enumerate(sorted_rows, start=1):
        row["global_rank"] = rank
        row["global_percentile"] = 100.0 * (n - rank) / max(n - 1, 1)
        row["in_top_1pct"] = int(rank <= top_ks[1])
        row["in_top_5pct"] = int(rank <= top_ks[5])
        row["in_top_10pct"] = int(rank <= top_ks[10])
    return rows


def read_ranking_scores(run_dir: Path, condition: str) -> List[Dict[str, object]]:
    rankings = read_csv(run_dir / "vog_rankings.csv")
    obs = observations_by_key(run_dir)
    by_obs: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for row in rankings:
        obs_id = int(row["observation_id"])
        sample_id = int(row["sample_id"])
        obs_row = obs.get((obs_id, sample_id), {})
        pred = obs_row.get("pred", "")
        target = int(row["target"])
        loss = parse_float(obs_row.get("loss"))
        pred_int = int(pred) if str(pred) != "" else ""
        by_obs[obs_id].append(
            {
                "condition": condition,
                "run_dir": str(run_dir),
                "score_source": "online_parameter_grad_norm_variance",
                "observation_id": obs_id,
                "epoch": int(row["epoch"]),
                "step": row.get("step", ""),
                "global_step": int(row["global_step"]),
                "sample_id": sample_id,
                "dataset_index": int(row["dataset_index"]),
                "target": target,
                "pred": pred_int,
                "is_misclassified": int(pred_int != target) if pred_int != "" else "",
                "loss": loss if loss is not None else "",
                "vog_score": float(row["vog_score"]),
            }
        )

    out: List[Dict[str, object]] = []
    for obs_rows in by_obs.values():
        out.extend(add_global_ranks(rank_rows_by_class(obs_rows)))
    return sorted(out, key=lambda row: (int(row["epoch"]), int(row["global_rank"])))


def layer_key(name: str) -> str:
    return name.rsplit(".", 1)[0] if "." in name else name


def vector_from_record(record: dict, scope: str, layer: str) -> torch.Tensor:
    if scope == "global" and isinstance(record.get("input_grad"), torch.Tensor):
        return record["input_grad"].detach().cpu().float().reshape(-1)

    grads = record.get("grads", {})
    parts: List[torch.Tensor] = []
    for name, grad in sorted(grads.items()):
        if not isinstance(grad, torch.Tensor):
            continue
        if scope == "layer" and layer_key(str(name)) != layer:
            continue
        parts.append(grad.detach().cpu().float().reshape(-1))
    if not parts:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(parts) if len(parts) > 1 else parts[0]


def raw_gradient_files(run_dir: Path) -> Dict[int, List[Path]]:
    files: Dict[int, List[Path]] = defaultdict(list)
    for path in sorted((run_dir / "gradients").glob("obs_*/sample_*.pt")):
        sample_id = int(path.stem.split("_", 1)[1])
        files[sample_id].append(path)
    return files


def read_raw_scores(run_dir: Path, condition: str, scope: str = "global", layer: str = "") -> List[Dict[str, object]]:
    files_by_sample = raw_gradient_files(run_dir)
    if not files_by_sample:
        raise ValueError(f"No raw gradient files found under {run_dir / 'gradients'}")

    by_obs: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for sample_id, files in sorted(files_by_sample.items()):
        count = 0
        mean: Optional[torch.Tensor] = None
        m2: Optional[torch.Tensor] = None
        score_source = "raw_parameter_gradient_vog"
        for path in files:
            record = torch.load(path, map_location="cpu")
            if scope == "global" and isinstance(record.get("input_grad"), torch.Tensor):
                score_source = "raw_input_gradient_vog"
            vec = vector_from_record(record, scope=scope, layer=layer)
            if vec.numel() == 0:
                continue
            count += 1
            if mean is None:
                mean = vec.clone()
                m2 = torch.zeros_like(vec)
            else:
                assert m2 is not None
                delta = vec - mean
                mean.add_(delta / count)
                delta2 = vec - mean
                m2.add_(delta * delta2)

            if count < 2:
                continue
            assert m2 is not None
            total_variance = float((m2 / count).sum().item())
            target = int(record["target"])
            pred = int(record["pred"])
            obs_id = int(record["observation_id"])
            by_obs[obs_id].append(
                {
                    "condition": condition,
                    "run_dir": str(run_dir),
                    "score_source": score_source,
                    "observation_id": obs_id,
                    "epoch": int(record["epoch"]),
                    "step": "" if record.get("step") is None else int(record["step"]),
                    "global_step": int(record["global_step"]),
                    "sample_id": int(record["sample_id"]),
                    "dataset_index": int(record["dataset_index"]),
                    "target": target,
                    "pred": pred,
                    "is_misclassified": int(pred != target),
                    "loss": float(record["loss"]),
                    "vog_score": math.sqrt(max(total_variance, 0.0)),
                }
            )

    out: List[Dict[str, object]] = []
    for obs_rows in by_obs.values():
        out.extend(add_global_ranks(rank_rows_by_class(obs_rows)))
    return sorted(out, key=lambda row: (int(row["epoch"]), int(row["global_rank"])))


def read_scores(run_dir: Path, condition: str, score_source: str, scope: str, layer: str) -> List[Dict[str, object]]:
    has_raw = bool(raw_gradient_files(run_dir))
    if score_source == "raw" or (score_source == "auto" and has_raw):
        return read_raw_scores(run_dir, condition, scope=scope, layer=layer)
    if not (run_dir / "vog_rankings.csv").exists():
        raise ValueError(f"No usable VoG ranking data in {run_dir}")
    return read_ranking_scores(run_dir, condition)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def bucket_summary(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, int, int], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["condition"]),
            str(row["score_source"]),
            int(row["epoch"]),
            int(row["class_decile"]),
        )
        grouped[key].append(row)

    out: List[Dict[str, object]] = []
    for (condition, score_source, epoch, decile), group in sorted(grouped.items()):
        losses = [float(row["loss"]) for row in group if parse_float(row.get("loss")) is not None]
        errors = [float(row["is_misclassified"]) for row in group if row.get("is_misclassified") != ""]
        out.append(
            {
                "condition": condition,
                "score_source": score_source,
                "epoch": epoch,
                "class_decile": decile,
                "num_samples": len(group),
                "mean_vog_score": mean([float(row["vog_score"]) for row in group]),
                "mean_class_z": mean([float(row["class_z"]) for row in group]),
                "mean_loss": mean(losses),
                "error_rate": mean(errors),
            }
        )
    return out


def late_sample_summary(rows: Sequence[Dict[str, object]], late_epochs: int) -> List[Dict[str, object]]:
    epochs = sorted({int(row["epoch"]) for row in rows})
    if not epochs:
        return []
    selected = set(epochs[-late_epochs:]) if late_epochs > 0 else set(epochs)
    grouped: Dict[Tuple[str, str, int], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        if int(row["epoch"]) in selected:
            grouped[(str(row["condition"]), str(row["score_source"]), int(row["sample_id"]))].append(row)

    out: List[Dict[str, object]] = []
    for (condition, score_source, sample_id), group in sorted(grouped.items()):
        n = len(group)
        top10_count = sum(int(row["in_top_10pct"]) for row in group)
        errors = [float(row["is_misclassified"]) for row in group if row.get("is_misclassified") != ""]
        loss_values = [float(row["loss"]) for row in group if parse_float(row.get("loss")) is not None]
        percentile_mean = mean([float(row["class_percentile"]) for row in group])
        top10_persistence = top10_count / max(n, 1)
        mis_rate = mean(errors)
        audit_score = 0.5 * (percentile_mean / 100.0) + 0.3 * top10_persistence + 0.2 * (0.0 if math.isnan(mis_rate) else mis_rate)
        first = group[0]
        out.append(
            {
                "condition": condition,
                "score_source": score_source,
                "sample_id": sample_id,
                "dataset_index": first["dataset_index"],
                "target": first["target"],
                "late_epoch_start": min(selected),
                "late_epoch_end": max(selected),
                "num_observations": n,
                "vog_mean_late": mean([float(row["vog_score"]) for row in group]),
                "class_z_mean_late": mean([float(row["class_z"]) for row in group]),
                "class_percentile_mean_late": percentile_mean,
                "top1_count_late": sum(int(row["in_top_1pct"]) for row in group),
                "top5_count_late": sum(int(row["in_top_5pct"]) for row in group),
                "top10_count_late": top10_count,
                "top10_persistence_late": top10_persistence,
                "misclassified_count_late": sum(errors),
                "misclassified_rate_late": mis_rate,
                "loss_mean_late": mean(loss_values),
                "audit_score": audit_score,
            }
        )
    return sorted(out, key=lambda row: (str(row["condition"]), -float(row["audit_score"])))


def analyze_runs(
    runs: Sequence[Tuple[str, Path]],
    output_dir: Path,
    score_source: str,
    scope: str,
    layer: str,
    late_epochs: int,
) -> Tuple[Path, Path, Path]:
    all_rows: List[Dict[str, object]] = []
    for condition, run_dir in runs:
        condition = canonical_condition(condition)
        rows = read_scores(run_dir, condition, score_source=score_source, scope=scope, layer=layer)
        print(f"{condition}: run_dir={run_dir} rows={len(rows)} source={rows[0]['score_source'] if rows else 'none'}")
        all_rows.extend(rows)

    if not all_rows:
        raise ValueError("No VoG rows were collected")

    per_sample_path = output_dir / "vog_paper_style_per_sample.csv"
    bucket_path = output_dir / "vog_paper_style_decile_summary.csv"
    late_path = output_dir / "vog_paper_style_late_sample_summary.csv"
    write_csv(all_rows, per_sample_path, PER_SAMPLE_FIELDS)
    write_csv(bucket_summary(all_rows), bucket_path, BUCKET_FIELDS)
    write_csv(late_sample_summary(all_rows, late_epochs=late_epochs), late_path, LATE_FIELDS)
    return per_sample_path, bucket_path, late_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper-style VoG ranking analysis.")
    parser.add_argument("--vog-log-root", type=Path, default=Path("../vog_logs"))
    parser.add_argument("--runs", nargs="*", default=[], help="Explicit CONDITION=RUN_DIR mappings.")
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--score-source", choices=["auto", "raw", "ranking"], default="auto")
    parser.add_argument("--scope", choices=["global", "layer"], default="global")
    parser.add_argument("--layer", default="", help="Layer name when --scope layer.")
    parser.add_argument("--late-epochs", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    vog_log_root = (script_dir / args.vog_log_root).resolve() if not args.vog_log_root.is_absolute() else args.vog_log_root
    output_dir = (script_dir / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    runs = resolve_runs(vog_log_root, args.conditions, args.runs)
    paths = analyze_runs(
        runs=runs,
        output_dir=output_dir,
        score_source=args.score_source,
        scope=args.scope,
        layer=args.layer,
        late_epochs=args.late_epochs,
    )
    print("outputs=" + " ".join(str(path) for path in paths))


if __name__ == "__main__":
    main()

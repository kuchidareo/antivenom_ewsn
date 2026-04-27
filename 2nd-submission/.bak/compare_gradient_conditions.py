import argparse
import copy
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from gradient_calculation_analysis import ACDAnalyzer, BackwardOperandCapture
from ml_running import PoisonDiskDataset, SimpleCNN, TransformConfig, sample_n_dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_train_dataset(data_root: Path, img_size: int, normalize: str, poison_frac: float, seed: int):
    tfm_cfg = TransformConfig(img_size=img_size, normalize=normalize)
    clean_ds = PoisonDiskDataset(variant_dir=data_root / "clean", split_name="train", tfm_cfg=tfm_cfg)

    target_n = len(clean_ds)
    if poison_frac <= 0.0:
        return sample_n_dataset(clean_ds, n=target_n, seed=seed + 1)

    blur_ds = PoisonDiskDataset(variant_dir=data_root / "blurring", split_name="train", tfm_cfg=tfm_cfg)
    poison_k = int(round(target_n * poison_frac))
    poison_k = max(0, min(poison_k, len(blur_ds)))
    clean_k = max(0, target_n - poison_k)

    clean_sampled = sample_n_dataset(clean_ds, n=clean_k, seed=seed + 1)
    blur_sampled = sample_n_dataset(blur_ds, n=poison_k, seed=seed + 2)
    return ConcatDataset([clean_sampled, blur_sampled])


def summarize_records(records):
    grouped = defaultdict(list)
    for r in records:
        grouped[(r["name"], r["kind"])].append(r)

    out = {}
    for key, rows in grouped.items():
        metrics = {}
        for metric in (
            "density",
            "fragmentation",
            "avg_run_length",
            "flat_partial_block_ratio",
            "flat_full_block_ratio",
            "md_partial_block_ratio",
            "md_full_block_ratio",
            "stability_jaccard",
        ):
            vals = [row[metric] for row in rows if row.get(metric) is not None]
            metrics[metric] = float(sum(vals) / len(vals)) if vals else None
        metrics["steps"] = len(rows)
        metrics["shape"] = rows[0]["shape"]
        out[key] = metrics
    return out


class LargeShapeACDAnalyzer(ACDAnalyzer):
    def __init__(
        self,
        *args,
        stability_max_elements=5_000_000,
        max_conv_positions=16,
        max_linear_features=512,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.stability_max_elements = stability_max_elements
        self.max_conv_positions = max_conv_positions
        self.max_linear_features = max_linear_features

    def _sample_l_positions(self, t: torch.Tensor, l_dim: int) -> torch.Tensor:
        l = t.shape[l_dim]
        if l <= self.max_conv_positions:
            return t
        idx = torch.linspace(0, l - 1, steps=self.max_conv_positions, device=t.device)
        idx = idx.round().long().unique(sorted=True)
        return torch.index_select(t, dim=l_dim, index=idx)

    def _sample_feature_dim(self, t: torch.Tensor, dim: int) -> torch.Tensor:
        n = t.shape[dim]
        if n <= self.max_linear_features:
            return t
        idx = torch.linspace(0, n - 1, steps=self.max_linear_features, device=t.device)
        idx = idx.round().long().unique(sorted=True)
        return torch.index_select(t, dim=dim, index=idx)

    def linear_dW_acd(self, x, grad_output):
        x = self._sample_feature_dim(x, dim=1)
        return super().linear_dW_acd(x, grad_output)

    def linear_dX_acd(self, weight, grad_output):
        weight = self._sample_feature_dim(weight, dim=1)
        return super().linear_dX_acd(weight, grad_output)

    def conv2d_dW_acd(self, module, x, grad_output):
        x_col = self._conv_unfold_input(module, x)                # [N, P, L]
        x_col = self._sample_l_positions(x_col, l_dim=2)
        n, p, l = x_col.shape
        dy = grad_output.reshape(n, grad_output.shape[1], -1)     # [N, Cout, L]
        dy = self._sample_l_positions(dy, l_dim=2)

        x_mask = self._active_mask(x_col).transpose(1, 2)         # [N, L, P]
        dy_mask = self._active_mask(dy).transpose(1, 2)           # [N, L, Cout]
        return dy_mask[:, :, :, None] & x_mask[:, :, None, :]

    def conv2d_dX_acd(self, module, grad_output):
        weight = module.weight.detach()
        cout = weight.shape[0]
        p = weight[0].numel()
        w_mask = self._active_mask(weight.reshape(cout, p))

        n = grad_output.shape[0]
        dy = grad_output.reshape(n, grad_output.shape[1], -1)
        dy = self._sample_l_positions(dy, l_dim=2)
        dy_mask = self._active_mask(dy).transpose(1, 2)
        return dy_mask[:, :, :, None] & w_mask[None, None, :, :]

    def _make_record(self, name, kind, acd_mask, step, prev_key):
        summary = self._summarize_mask(acd_mask)

        prev = self.prev_masks.get(prev_key, None)
        current_cpu = None
        stability = None
        if acd_mask.numel() <= self.stability_max_elements:
            current_cpu = acd_mask.detach().cpu()
            stability = self._jaccard(prev, current_cpu) if prev is not None else None
            self.prev_masks[prev_key] = current_cpu
        else:
            self.prev_masks.pop(prev_key, None)

        return {
            "step": step,
            "name": name,
            "kind": kind,
            "shape": tuple(acd_mask.shape),
            "stability_jaccard": stability,
            **summary,
        }


def print_condition_summary(name: str, summary: dict) -> None:
    print(f"\n===== {name} =====")
    for layer, kind in sorted(summary.keys()):
        s = summary[(layer, kind)]
        print(
            f"{layer:>5s} | {kind:>9s} | shape={s['shape']} | "
            f"density={s['density']:.6f} | frag={s['fragmentation']:.6f} | "
            f"avg_run={s['avg_run_length']:.2f} | flat_partial={s['flat_partial_block_ratio']:.4f} | "
            f"md_partial={s['md_partial_block_ratio']:.4f} | stability={s['stability_jaccard']}"
        )


def print_delta_summary(clean_summary: dict, blur_summary: dict) -> None:
    print("\n===== BLURRING30 - CLEAN DELTA =====")
    for key in sorted(clean_summary.keys()):
        if key not in blur_summary:
            continue
        c = clean_summary[key]
        b = blur_summary[key]
        print(
            f"{key[0]:>5s} | {key[1]:>9s} | "
            f"density_delta={b['density'] - c['density']:+.6f} | "
            f"frag_delta={b['fragmentation'] - c['fragmentation']:+.6f} | "
            f"avg_run_delta={b['avg_run_length'] - c['avg_run_length']:+.2f} | "
            f"flat_partial_delta={b['flat_partial_block_ratio'] - c['flat_partial_block_ratio']:+.4f} | "
            f"stability_delta="
            f"{(b['stability_jaccard'] - c['stability_jaccard']) if (b['stability_jaccard'] is not None and c['stability_jaccard'] is not None) else None}"
        )


def run_condition(
    condition_name: str,
    data_root: Path,
    base_model_state: dict,
    device: str,
    steps: int,
    batch_size: int,
    img_size: int,
    normalize: str,
    poison_frac: float,
    seed: int,
):
    dataset = build_train_dataset(
        data_root=data_root,
        img_size=img_size,
        normalize=normalize,
        poison_frac=poison_frac,
        seed=seed,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = SimpleCNN(n_classes=6, img_size=img_size).to(device)
    model.load_state_dict(copy.deepcopy(base_model_state))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    capture = BackwardOperandCapture(model)
    analyzer = LargeShapeACDAnalyzer(
        eps=1e-8,
        flat_block_size=64,
        md_block_sizes={
            3: (4, 8, 16),
            4: (2, 8, 8, 16),
        },
    )

    step = 0
    for x, target in loader:
        if step >= steps:
            break
        print(f"{condition_name}: step {step + 1}/{steps}", flush=True)
        x = x.to(device)
        target = target.to(device)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        capture.clear()

        logits = model(x)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        for module_name, saved in capture.saved.items():
            grad_output = saved["output"].grad.detach()
            analyzer.analyze_module(
                name=module_name,
                module=saved["module"],
                x=saved["input"],
                grad_output=grad_output,
                step=step,
            )
        step += 1

    capture.close()
    summary = summarize_records(analyzer.get_records())
    print_condition_summary(condition_name, summary)
    return {
        "condition": condition_name,
        "steps_run": step,
        "dataset_len": len(dataset),
        "summary": summary,
    }


def sanitize_for_json(summary: dict) -> dict:
    out = {}
    for (layer, kind), values in summary.items():
        out[f"{layer}:{kind}"] = values
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="../data")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--normalize", type=str, default="0.5")
    p.add_argument("--poison-frac", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-json", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    data_root = Path(args.data_root)
    device = args.device

    base_model = SimpleCNN(n_classes=6, img_size=args.img_size)
    base_model_state = copy.deepcopy(base_model.state_dict())

    clean = run_condition(
        condition_name="clean",
        data_root=data_root,
        base_model_state=base_model_state,
        device=device,
        steps=args.steps,
        batch_size=args.batch_size,
        img_size=args.img_size,
        normalize=args.normalize,
        poison_frac=0.0,
        seed=args.seed,
    )
    blur = run_condition(
        condition_name="blurring30",
        data_root=data_root,
        base_model_state=base_model_state,
        device=device,
        steps=args.steps,
        batch_size=args.batch_size,
        img_size=args.img_size,
        normalize=args.normalize,
        poison_frac=args.poison_frac,
        seed=args.seed,
    )

    print_delta_summary(clean["summary"], blur["summary"])

    if args.out_json:
        out = {
            "config": {
                "data_root": str(data_root),
                "device": device,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "img_size": args.img_size,
                "normalize": args.normalize,
                "poison_frac": args.poison_frac,
                "seed": args.seed,
            },
            "clean": {
                "steps_run": clean["steps_run"],
                "dataset_len": clean["dataset_len"],
                "summary": sanitize_for_json(clean["summary"]),
            },
            "blurring30": {
                "steps_run": blur["steps_run"],
                "dataset_len": blur["dataset_len"],
                "summary": sanitize_for_json(blur["summary"]),
            },
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()

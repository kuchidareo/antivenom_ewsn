import argparse
import copy
import json
import random
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity
from torch.utils.data import ConcatDataset, DataLoader

from ml_running import PoisonDiskDataset, SimpleCNN, TransformConfig, sample_n_dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_condition_dataset(
    data_root: Path,
    img_size: int,
    normalize: str,
    poison_frac: float,
    seed: int,
):
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


# =========================================================
# 1. Layout utilities
# =========================================================
def tensor_layout_info(t: torch.Tensor):
    if t is None:
        return None

    info = {
        "shape": tuple(t.shape),
        "stride": tuple(t.stride()),
        "is_contiguous": t.is_contiguous(),
    }

    # channels_last only makes sense for 4D tensors
    if t.dim() == 4:
        info["is_channels_last"] = t.is_contiguous(memory_format=torch.channels_last)
    else:
        info["is_channels_last"] = None

    return info


def layout_signature(info):
    if info is None:
        return None
    return (
        info["shape"],
        info["stride"],
        info["is_contiguous"],
        info["is_channels_last"],
    )


# =========================================================
# 2. Capture forward output + backward grad layout
# =========================================================
class LayoutCapture:
    def __init__(self, model):
        self.saved = {}
        self.handles = []

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                h = module.register_forward_hook(self._make_hook(name))
                self.handles.append(h)

    def _make_hook(self, name):
        def hook(module, inputs, output):
            output.retain_grad()
            self.saved[name] = {
                "module": module,
                "input_info": tensor_layout_info(inputs[0]),
                "output": output,
                "output_info": tensor_layout_info(output),
            }
        return hook

    def finalize_backward(self):
        for name, rec in self.saved.items():
            grad = rec["output"].grad
            rec["grad_output_info"] = tensor_layout_info(grad)

            # simple "layout conflict" proxy:
            # output layout and grad_output layout differ
            rec["output_grad_layout_changed"] = (
                layout_signature(rec["output_info"]) != layout_signature(rec["grad_output_info"])
            )

    def clear(self):
        self.saved.clear()

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []


# =========================================================
# 3. Backward graph traversal
# =========================================================
def traverse_backward_graph(loss):
    """
    Traverse autograd graph from loss.grad_fn and collect node names.
    """
    root = loss.grad_fn
    if root is None:
        return []

    nodes = []
    visited = set()
    q = deque([root])

    while q:
        fn = q.popleft()
        if fn is None:
            continue

        fn_id = id(fn)
        if fn_id in visited:
            continue
        visited.add(fn_id)

        name = type(fn).__name__
        nodes.append(name)

        if hasattr(fn, "next_functions"):
            for nxt, _ in fn.next_functions:
                if nxt is not None:
                    q.append(nxt)

    return nodes


def summarize_backward_graph(node_names):
    counts = defaultdict(int)
    for n in node_names:
        counts[n] += 1

    interesting_keywords = [
        "Permute", "Transpose", "View", "Reshape", "Clone", "Copy",
        "Mm", "Addmm", "Bmm", "Convolution", "Relu", "AccumulateGrad"
    ]

    interesting = {}
    for k, v in counts.items():
        if any(word.lower() in k.lower() for word in interesting_keywords):
            interesting[k] = v

    return dict(counts), interesting


# =========================================================
# 4. Profiler helpers
# =========================================================
def profile_one_step(model, x, y, criterion, optimizer, device):
    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False
    ) as prof:
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

    return prof


def extract_layout_related_ops(prof):
    """
    Extract layout-conversion-ish operators from profiler.
    """
    keywords = [
        "contiguous",
        "clone",
        "copy",
        "permute",
        "transpose",
        "reshape",
        "view",
        "_to_copy",
    ]

    rows = []
    for evt in prof.key_averages():
        key = evt.key
        if any(k in key.lower() for k in keywords):
            rows.append({
                "key": key,
                "count": getattr(evt, "count", None),
                "self_cpu_time_total": getattr(evt, "self_cpu_time_total", None),
                "cpu_time_total": getattr(evt, "cpu_time_total", None),
                "self_cuda_time_total": getattr(evt, "self_cuda_time_total", None),
                "cuda_time_total": getattr(evt, "cuda_time_total", None),
            })
    return rows


def aggregate_layout_op_time(rows):
    total_cpu = 0.0
    total_cuda = 0.0
    total_count = 0
    for r in rows:
        total_count += int(r["count"] or 0)
        total_cpu += float(r["self_cpu_time_total"] or 0.0)
        total_cuda += float(r["self_cuda_time_total"] or 0.0)
    return {
        "layout_op_count": total_count,
        "layout_self_cpu_time_total": total_cpu,
        "layout_self_cuda_time_total": total_cuda,
    }


def collect_one_batch(loader, step):
    for batch_idx, batch in enumerate(loader):
        if batch_idx == step:
            return batch
    return None


# =========================================================
# 5. One condition runner
# =========================================================
def run_condition(model, condition_name, loader, steps=5, device="cpu"):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    capture = LayoutCapture(model)

    per_step = []

    for step in range(steps):
        capture.clear()

        batch = collect_one_batch(loader, step)
        if batch is None:
            break
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        # profiled pass
        prof = profile_one_step(model, x, y, criterion, optimizer, device)

        # second pass for graph/capture consistency
        optimizer.zero_grad(set_to_none=True)
        capture.clear()
        logits = model(x)
        loss = criterion(logits, y)
        node_names = traverse_backward_graph(loss)
        loss.backward()
        optimizer.step()

        capture.finalize_backward()

        all_counts, interesting_counts = summarize_backward_graph(node_names)
        layout_rows = extract_layout_related_ops(prof)
        layout_op_summary = aggregate_layout_op_time(layout_rows)

        changed_layers = {}
        for name, rec in capture.saved.items():
            changed_layers[name] = {
                "input_info": rec["input_info"],
                "output_info": rec["output_info"],
                "grad_output_info": rec["grad_output_info"],
                "output_grad_layout_changed": rec["output_grad_layout_changed"],
            }

        per_step.append({
            "condition": condition_name,
            "step": step,
            "backward_graph_interesting_counts": interesting_counts,
            "layout_related_profiler_rows": layout_rows,
            "layout_op_summary": layout_op_summary,
            "layer_layouts": changed_layers,
        })

    capture.close()
    return per_step


# =========================================================
# 6. Compare conditions
# =========================================================
def summarize_condition_results(results):
    summary = {
        "avg_layout_op_count": 0.0,
        "avg_layout_self_cpu_time_total": 0.0,
        "avg_layout_self_cuda_time_total": 0.0,
        "layer_output_grad_layout_changed_rate": defaultdict(float),
        "backward_node_counts": defaultdict(float),
        "num_steps": len(results),
    }

    n = len(results)
    for step_rec in results:
        los = step_rec["layout_op_summary"]
        summary["avg_layout_op_count"] += los["layout_op_count"]
        summary["avg_layout_self_cpu_time_total"] += los["layout_self_cpu_time_total"]
        summary["avg_layout_self_cuda_time_total"] += los["layout_self_cuda_time_total"]

        for lname, lrec in step_rec["layer_layouts"].items():
            summary["layer_output_grad_layout_changed_rate"][lname] += float(
                lrec["output_grad_layout_changed"]
            )

        for k, v in step_rec["backward_graph_interesting_counts"].items():
            summary["backward_node_counts"][k] += v

    if n > 0:
        summary["avg_layout_op_count"] /= n
        summary["avg_layout_self_cpu_time_total"] /= n
        summary["avg_layout_self_cuda_time_total"] /= n

        for lname in list(summary["layer_output_grad_layout_changed_rate"].keys()):
            summary["layer_output_grad_layout_changed_rate"][lname] /= n

        for k in list(summary["backward_node_counts"].keys()):
            summary["backward_node_counts"][k] /= n

    return summary


def sanitize_for_json(obj):
    if isinstance(obj, defaultdict):
        obj = dict(obj)
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    return obj


def print_summary(name, summary):
    print(f"\n===== {name} SUMMARY =====")
    print("avg_layout_op_count:", summary["avg_layout_op_count"])
    print("avg_layout_self_cpu_time_total:", summary["avg_layout_self_cpu_time_total"])
    print("avg_layout_self_cuda_time_total:", summary["avg_layout_self_cuda_time_total"])

    print("\nlayer_output_grad_layout_changed_rate")
    for lname, v in sorted(summary["layer_output_grad_layout_changed_rate"].items()):
        print(f"  {lname}: {v:.3f}")

    print("\nbackward_node_counts (interesting)")
    for k, v in sorted(summary["backward_node_counts"].items()):
        print(f"  {k}: {v:.3f}")


def print_delta(clean_summary, blur_summary):
    print("\n===== BLUR - CLEAN DELTA =====")
    print(
        "layout_op_count_delta:",
        blur_summary["avg_layout_op_count"] - clean_summary["avg_layout_op_count"]
    )
    print(
        "layout_self_cpu_time_delta:",
        blur_summary["avg_layout_self_cpu_time_total"] - clean_summary["avg_layout_self_cpu_time_total"]
    )
    print(
        "layout_self_cuda_time_delta:",
        blur_summary["avg_layout_self_cuda_time_total"] - clean_summary["avg_layout_self_cuda_time_total"]
    )

    print("\nlayer_output_grad_layout_changed_rate_delta")
    layer_names = sorted(set(clean_summary["layer_output_grad_layout_changed_rate"].keys()) |
                         set(blur_summary["layer_output_grad_layout_changed_rate"].keys()))
    for lname in layer_names:
        c = clean_summary["layer_output_grad_layout_changed_rate"].get(lname, 0.0)
        b = blur_summary["layer_output_grad_layout_changed_rate"].get(lname, 0.0)
        print(f"  {lname}: {b - c:+.3f}")

    print("\nbackward_node_count_delta (interesting)")
    node_names = sorted(set(clean_summary["backward_node_counts"].keys()) |
                        set(blur_summary["backward_node_counts"].keys()))
    for n in node_names:
        c = clean_summary["backward_node_counts"].get(n, 0.0)
        b = blur_summary["backward_node_counts"].get(n, 0.0)
        d = b - c
        if abs(d) > 1e-9:
            print(f"  {n}: {d:+.3f}")


# =========================================================
# 7. Main
# =========================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="../data")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--normalize", type=str, default="0.5")
    p.add_argument("--poison-frac", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-json", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    clean_ds = build_condition_dataset(
        data_root=Path(args.data_root),
        img_size=args.img_size,
        normalize=args.normalize,
        poison_frac=0.0,
        seed=args.seed,
    )
    blur_ds = build_condition_dataset(
        data_root=Path(args.data_root),
        img_size=args.img_size,
        normalize=args.normalize,
        poison_frac=args.poison_frac,
        seed=args.seed,
    )

    clean_loader = DataLoader(clean_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    blur_loader = DataLoader(blur_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model_clean = SimpleCNN(n_classes=6, img_size=args.img_size).to(device)
    model_blur = SimpleCNN(n_classes=6, img_size=args.img_size).to(device)

    model_blur.load_state_dict(copy.deepcopy(model_clean.state_dict()))

    clean_results = run_condition(
        model_clean,
        condition_name="CLEAN",
        loader=clean_loader,
        steps=args.steps,
        device=device,
    )

    blur_results = run_condition(
        model_blur,
        condition_name="BLURRING30",
        loader=blur_loader,
        steps=args.steps,
        device=device,
    )

    clean_summary = summarize_condition_results(clean_results)
    blur_summary = summarize_condition_results(blur_results)

    print_summary("CLEAN", clean_summary)
    print_summary("BLUR", blur_summary)
    print_delta(clean_summary, blur_summary)

    if args.out_json:
        out = {
            "config": {
                "data_root": str(Path(args.data_root)),
                "device": device,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "img_size": args.img_size,
                "normalize": args.normalize,
                "poison_frac": args.poison_frac,
                "seed": args.seed,
                "model": "ml_running.SimpleCNN",
                "blur_source": "pre-generated ../data/blurring from data_preparing.py GaussianBlur(radius=12.0)",
            },
            "clean": {
                "steps_run": len(clean_results),
                "summary": sanitize_for_json(clean_summary),
            },
            "blurring30": {
                "steps_run": len(blur_results),
                "summary": sanitize_for_json(blur_summary),
            },
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()

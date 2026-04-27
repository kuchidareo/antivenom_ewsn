from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch


GROUPS = [
    ("forward_spatial", "Forward Spatial", "cividis", "Spatial Position"),
    ("backward_spatial", "Backward Spatial", "cividis", "Spatial Position"),
    ("forward_cacheline", "Forward Cache-Line", "viridis", "Cache-Line Block Index"),
    ("forward_page", "Forward Page", "magma", "Page Block Index"),
    ("backward_cacheline", "Backward Cache-Line", "viridis", "Cache-Line Block Index"),
    ("backward_page", "Backward Page", "magma", "Page Block Index"),
]


def _load_pt_paths(path_str: str) -> list[Path]:
    path = Path(path_str)
    if path.is_dir():
        return sorted(path.glob("*.pt"))
    return [path]


def _average_payloads(paths: list[Path]) -> dict:
    if not paths:
        raise FileNotFoundError("No .pt occupancy snapshots found.")

    payloads = [torch.load(path, map_location="cpu") for path in paths]
    out: dict = {"meta": {"num_snapshots": len(paths), "source_paths": [str(path) for path in paths]}}

    for key, _, _, _ in GROUPS:
        grouped: dict[str, list[torch.Tensor]] = defaultdict(list)
        for payload in payloads:
            for layer, tensor in payload.get(key, {}).items():
                grouped[str(layer)].append(tensor.detach().float().cpu())

        out[key] = {}
        for layer, tensors in grouped.items():
            ref_shape = tensors[0].shape
            same_shape = [tensor for tensor in tensors if tensor.shape == ref_shape]
            if not same_shape:
                continue
            stacked = torch.stack(same_shape, dim=0)
            out[key][layer] = stacked.mean(dim=0)

    return out


def _layer_names(clean_payload: dict, poison_payload: dict, key: str) -> list[str]:
    names: list[str] = []
    for source in (clean_payload.get(key, {}), poison_payload.get(key, {})):
        for name in source.keys():
            if name not in names:
                names.append(name)
    return names


def _layer_vector(payload: dict, key: str, layer: str) -> torch.Tensor:
    layer_dict = payload.get(key, {})
    if layer not in layer_dict:
        return torch.zeros(1, dtype=torch.float32)
    return layer_dict[layer].detach().float().cpu().reshape(-1)


def _layer_image(payload: dict, key: str, layer: str) -> torch.Tensor:
    layer_dict = payload.get(key, {})
    if layer not in layer_dict:
        return torch.zeros((1, 1), dtype=torch.float32)
    tensor = layer_dict[layer].detach().float().cpu()
    if tensor.dim() == 0:
        return tensor.view(1, 1)
    if tensor.dim() == 1:
        return tensor.unsqueeze(0)
    return tensor


def _pair_limits(clean_vec: torch.Tensor, poison_vec: torch.Tensor) -> tuple[float, float]:
    combined = torch.cat([clean_vec, poison_vec], dim=0)
    vmin = float(combined.min().item()) if combined.numel() else 0.0
    vmax = float(combined.max().item()) if combined.numel() else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def _spread_score(vec: torch.Tensor) -> float:
    flat = vec.detach().float().reshape(-1)
    if flat.numel() == 0:
        return 0.0
    return float((flat > 0).float().mean().item())


def _save_spread_csv(clean_payload: dict, poison_payload: dict, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for key, _, _, _ in GROUPS:
        if "spatial" in key:
            continue
        for layer in _layer_names(clean_payload, poison_payload, key):
            clean_vec = _layer_vector(clean_payload, key, layer)
            poison_vec = _layer_vector(poison_payload, key, layer)
            rows.append(
                {
                    "group": key,
                    "layer": layer,
                    "clean_spread": _spread_score(clean_vec),
                    "poison_spread": _spread_score(poison_vec),
                    "clean_mean": float(clean_vec.mean().item()) if clean_vec.numel() else 0.0,
                    "poison_mean": float(poison_vec.mean().item()) if poison_vec.numel() else 0.0,
                    "clean_blocks": int(clean_vec.numel()),
                    "poison_blocks": int(poison_vec.numel()),
                }
            )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group",
                "layer",
                "clean_spread",
                "poison_spread",
                "clean_mean",
                "poison_mean",
                "clean_blocks",
                "poison_blocks",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", required=True, help="Clean .pt file or directory of .pt occupancy snapshots.")
    parser.add_argument("--poison", required=True, help="Poisoned .pt file or directory of .pt occupancy snapshots.")
    parser.add_argument("--out", default="2nd-submission/figures/occupancy_compare_heatmap.png")
    parser.add_argument("--spread-csv", default="2nd-submission/figures/occupancy_spread_summary.csv")
    args = parser.parse_args()

    clean_paths = _load_pt_paths(args.clean)
    poison_paths = _load_pt_paths(args.poison)
    clean_payload = _average_payloads(clean_paths)
    poison_payload = _average_payloads(poison_paths)

    max_layers = max(len(_layer_names(clean_payload, poison_payload, key)) for key, _, _, _ in GROUPS)
    fig, axes = plt.subplots(len(GROUPS) * 2, max_layers, figsize=(3.2 * max_layers, 2.2 * len(GROUPS) * 2))
    if max_layers == 1:
        axes = axes.reshape(len(GROUPS) * 2, 1)

    for group_idx, (key, title, cmap, xlabel) in enumerate(GROUPS):
        layer_names = _layer_names(clean_payload, poison_payload, key)
        clean_row = group_idx * 2
        poison_row = clean_row + 1

        for col in range(max_layers):
            ax_clean = axes[clean_row, col]
            ax_poison = axes[poison_row, col]

            if col >= len(layer_names):
                ax_clean.axis("off")
                ax_poison.axis("off")
                continue

            layer = layer_names[col]
            if "spatial" in key:
                clean_img = _layer_image(clean_payload, key, layer)
                poison_img = _layer_image(poison_payload, key, layer)
                vmin, vmax = _pair_limits(clean_img.reshape(-1), poison_img.reshape(-1))
                im_clean = ax_clean.imshow(
                    clean_img.numpy(),
                    aspect="auto",
                    interpolation="nearest",
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )
                ax_poison.imshow(
                    poison_img.numpy(),
                    aspect="auto",
                    interpolation="nearest",
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )
                title_text = layer
            else:
                clean_vec = _layer_vector(clean_payload, key, layer)
                poison_vec = _layer_vector(poison_payload, key, layer)
                vmin, vmax = _pair_limits(clean_vec, poison_vec)
                clean_spread = _spread_score(clean_vec)
                poison_spread = _spread_score(poison_vec)
                im_clean = ax_clean.imshow(
                    clean_vec.unsqueeze(0).numpy(),
                    aspect="auto",
                    interpolation="nearest",
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )
                ax_poison.imshow(
                    poison_vec.unsqueeze(0).numpy(),
                    aspect="auto",
                    interpolation="nearest",
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )
                title_text = f"{layer}\nC={clean_spread:.3f} P={poison_spread:.3f}"

            ax_clean.set_title(title_text)
            ax_clean.set_yticks([0])
            ax_clean.set_yticklabels(["Clean"])
            ax_poison.set_yticks([0])
            ax_poison.set_yticklabels(["Blur30"])
            ax_clean.set_xticks([])
            ax_poison.set_xticks([])

            if col == 0:
                ax_clean.set_ylabel(title)
                ax_poison.set_ylabel(title)

            cbar = fig.colorbar(im_clean, ax=[ax_clean, ax_poison], fraction=0.03, pad=0.02)
            cbar.ax.set_ylabel("Magnitude" if "spatial" in key else "Avg Active Count", rotation=270, labelpad=12)

        for col in range(len(layer_names)):
            axes[poison_row, col].set_xlabel(xlabel)

    fig.suptitle(
        "Occupancy Comparison by Layer\n"
        f"Average of {clean_payload['meta']['num_snapshots']} clean snapshots vs "
        f"{poison_payload['meta']['num_snapshots']} blurring snapshots",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")

    spread_csv = Path(args.spread_csv)
    _save_spread_csv(clean_payload, poison_payload, spread_csv)
    print(f"saved_plot={out_path}")
    print(f"saved_spread_csv={spread_csv}")


if __name__ == "__main__":
    main()

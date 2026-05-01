from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent
INPUT_CSV = (
    ROOT_DIR
    / "results"
    / "focused"
    / "{scenario}__c11_memory100_epoch_ot_rows.csv"
)
OUT_DIR = ROOT_DIR / "results" / "paper_selected"

# Edit these variables for paper styling.
FIGSIZE = (2.6, 1.5)
DPI = 300
LINE_WIDTH = 2.1
MARKER_SIZE = 4.2
RANGE_ALPHA = 0.18
SHOW_MARKERS = True
SHOW_GRID = True

CLEAN_COLOR = "#4C78A8"
REFERENCE_CLEAN_COLOR = "#59A14F"
POISONED_COLOR = "#E67D7E"
CLEAN_MARKER = "o"
REFERENCE_CLEAN_MARKER = "^"
POISONED_MARKER = "s"

TITLE_FONT_SIZE = 10
LABEL_FONT_SIZE = 8
TICK_FONT_SIZE = 8
LEGEND_FONT_SIZE = 8
LEGEND_ONLY_FIGSIZE = (2.9, 0.35)
LEGEND_ONLY_OUTPUT = "legend_clean_poisoned.png"
LEGEND_GROUPS = ("base_clean", "augmentation_clean", "poisoned")

SCENARIOS = {
    "adamw_weight_decay": {
        "title": "AdamW Weight Decay",
        "output": "adamw_weight_decay__c11_memory100_mean_range.png",
    },
    "model_pruning": {
        "title": "Model Pruning",
        "output": "model_pruning__c11_memory100_mean_range.png",
    },
    "data_augmentation": {
        "title": "Data Augmentation",
        "output": "data_augmentation__c11_memory100_mean_range.png",
        "plot_groups": {
            "baseclean": "base_clean",
            "data_augmentation/clean": "augmentation_clean",
            "baseblurring": "poisoned",
        },
    },
    "ood": {
        "title": "OOD",
        "output": "ood__c11_memory100_mean_range.png",
        "plot_groups": {
            "baseclean": "base_clean",
            "OOD_data_training/clean": "ood_clean",
            "baseblurring": "poisoned",
        },
    },
}

CLASS_STYLE = {
    "clean": {
        "label": "Clean",
        "color": CLEAN_COLOR,
        "marker": CLEAN_MARKER,
    },
    "base_clean": {
        "label": "Clean",
        "color": CLEAN_COLOR,
        "marker": CLEAN_MARKER,
    },
    "augmentation_clean": {
        "label": "Augmentation/OOD",
        "color": REFERENCE_CLEAN_COLOR,
        "marker": REFERENCE_CLEAN_MARKER,
    },
    "ood_clean": {
        "label": "Augmentation/OOD",
        "color": REFERENCE_CLEAN_COLOR,
        "marker": REFERENCE_CLEAN_MARKER,
    },
    "poisoned": {
        "label": "Blurring",
        "color": POISONED_COLOR,
        "marker": POISONED_MARKER,
    },
}
DEFAULT_PLOT_ORDER = ("clean", "poisoned")


def style_axis(ax: plt.Axes) -> None:
    if SHOW_GRID:
        ax.grid(axis="y", alpha=0.22, linewidth=0.8)
        ax.grid(axis="x", alpha=0.10, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)


def plot_mean_range(df: pd.DataFrame, scenario: str, config: dict[str, object]) -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)

    plot_groups = config.get("plot_groups")
    if plot_groups:
        plot_df = df.copy()
        plot_df["plot_group"] = plot_df["target_group"].map(plot_groups)
        plot_order = tuple(plot_groups.values())
    else:
        plot_df = df.copy()
        plot_df["plot_group"] = plot_df["classification_label"]
        plot_order = DEFAULT_PLOT_ORDER

    for plot_group in plot_order:
        group_df = plot_df.loc[plot_df["plot_group"] == plot_group]
        if group_df.empty:
            continue

        trend = (
            group_df.groupby("round", as_index=False)["ot_distance"]
            .agg(["mean", "min", "max"])
            .reset_index()
            .sort_values("round")
        )
        style = CLASS_STYLE[plot_group]
        marker = style["marker"] if SHOW_MARKERS else None
        x = trend["round"].to_numpy(dtype=float) + 1.0
        y = trend["mean"].to_numpy(dtype=float)
        y_min = trend["min"].to_numpy(dtype=float)
        y_max = trend["max"].to_numpy(dtype=float)

        ax.plot(
            x,
            y,
            color=style["color"],
            marker=marker,
            markersize=MARKER_SIZE,
            linewidth=LINE_WIDTH,
            label=style["label"],
        )
        ax.fill_between(
            x,
            y_min,
            y_max,
            color=style["color"],
            alpha=RANGE_ALPHA,
            linewidth=0,
        )

    style_axis(ax)
    # ax.set_title(config["title"], fontsize=TITLE_FONT_SIZE)
    ax.set_xlabel("Round", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("Magnitude score W", fontsize=LABEL_FONT_SIZE)
    # ax.legend(frameon=False, fontsize=LEGEND_FONT_SIZE, loc="best")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / config["output"]
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return out_path


def plot_legend_only() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=LEGEND_ONLY_FIGSIZE)
    ax.axis("off")

    handles = []
    labels = []
    for class_name in LEGEND_GROUPS:
        style = CLASS_STYLE[class_name]
        marker = style["marker"] if SHOW_MARKERS else None
        handle = ax.plot(
            [],
            [],
            color=style["color"],
            marker=marker,
            markersize=MARKER_SIZE,
            linewidth=LINE_WIDTH,
        )[0]
        handles.append(handle)
        labels.append(style["label"])

    fig.legend(
        handles,
        labels,
        loc="center",
        ncol=2,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=1.8,
        columnspacing=1.2,
    )
    out_path = OUT_DIR / LEGEND_ONLY_OUTPUT
    fig.savefig(out_path, dpi=DPI, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out_path


def main() -> None:
    saved = []
    for scenario, config in SCENARIOS.items():
        csv_path = Path(str(INPUT_CSV).format(scenario=scenario))
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing input CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        saved.append(plot_mean_range(df, scenario, config))
    saved.append(plot_legend_only())

    for path in saved:
        print(path)


if __name__ == "__main__":
    main()

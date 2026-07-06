"""Produce the H1/H2 evidence figures from results/controlled_experiment.csv.

Figure 1: accuracy vs mismatch level, by method, faceted by stationarity  (H1, H2)
Figure 2: spatial generalisation gap vs mismatch level, by method          (H1, H2)
Figure 3: measured divergence (MMD) vs out-of-region accuracy               (validates section 6.3 metrics)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]

# Palette (dataviz skill reference instance, light mode)
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

METHOD_COLORS = {
    "supervised_only": "#2a78d6",     # categorical slot 1 (blue)
    "self_training": "#1baf7a",       # categorical slot 2 (aqua)
    "label_propagation": "#eda100",   # categorical slot 3 (yellow)
}
METHOD_LABELS = {
    "supervised_only": "Supervised-only",
    "self_training": "Self-training",
    "label_propagation": "Label propagation",
}


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)


def plot_accuracy_vs_mismatch(df: pd.DataFrame, out_path: Path):
    stat_levels = sorted(df["nonstationarity_strength"].unique())
    fig, axes = plt.subplots(1, len(stat_levels), figsize=(11, 4.5), sharey=True, facecolor=SURFACE)
    if len(stat_levels) == 1:
        axes = [axes]

    for ax, stat in zip(axes, stat_levels):
        sub = df[df["nonstationarity_strength"] == stat]
        for method in METHOD_COLORS:
            g = sub[sub["method"] == method].groupby("mismatch_alpha")["accuracy_out_region"].mean()
            ax.plot(g.index, g.values, marker="o", markersize=6, linewidth=2,
                     color=METHOD_COLORS[method], label=METHOD_LABELS[method], zorder=3)
        _style_axes(ax)
        title = "Stationary label process" if stat == 0 else f"Non-stationary (strength={stat})"
        ax.set_title(title, color=INK_PRIMARY, fontsize=11, loc="left")
        ax.set_xlabel("Mismatch level (mismatch_alpha)", color=INK_SECONDARY, fontsize=10)

    axes[0].set_ylabel("Out-of-region accuracy", color=INK_SECONDARY, fontsize=10)
    axes[-1].legend(frameon=False, fontsize=9, loc="lower left", labelcolor=INK_SECONDARY)
    fig.suptitle("H1/H2: SSL performance degrades as labelled-unlabelled mismatch increases",
                 color=INK_PRIMARY, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_spatial_gap_vs_mismatch(df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
    for method in METHOD_COLORS:
        g = df[df["method"] == method].groupby("mismatch_alpha")["spatial_generalisation_gap"].mean()
        ax.plot(g.index, g.values, marker="o", markersize=6, linewidth=2,
                 color=METHOD_COLORS[method], label=METHOD_LABELS[method], zorder=3)
    ax.axhline(0, color=BASELINE, linewidth=1, zorder=1)
    _style_axes(ax)
    ax.set_xlabel("Mismatch level (mismatch_alpha)", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("Spatial generalisation gap (in-region − out-of-region accuracy)",
                   color=INK_SECONDARY, fontsize=10)
    ax.set_title("H1: models overfit the labelled region as mismatch grows",
                 color=INK_PRIMARY, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_divergence_vs_accuracy(df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
    for method in METHOD_COLORS:
        sub = df[df["method"] == method]
        ax.scatter(sub["mmd"], sub["accuracy_out_region"], s=22, alpha=0.6,
                    color=METHOD_COLORS[method], label=METHOD_LABELS[method], zorder=3,
                    edgecolors="none")
    _style_axes(ax)
    ax.set_xlabel("Measured mismatch (MMD, labelled vs unlabelled)", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("Out-of-region accuracy", color=INK_SECONDARY, fontsize=10)
    ax.set_title("Section 6.3 validation: higher measured MMD tracks lower accuracy",
                 color=INK_PRIMARY, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower left", labelcolor=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main(csv_path: str | None = None):
    csv_path = Path(csv_path) if csv_path else REPO_ROOT / "results" / "controlled_experiment.csv"
    df = pd.read_csv(csv_path)
    fig_dir = REPO_ROOT / "figures"
    fig_dir.mkdir(exist_ok=True)

    plot_accuracy_vs_mismatch(df, fig_dir / "fig1_accuracy_vs_mismatch.png")
    plot_spatial_gap_vs_mismatch(df, fig_dir / "fig2_spatial_gap_vs_mismatch.png")
    plot_divergence_vs_accuracy(df, fig_dir / "fig3_divergence_vs_accuracy.png")
    print(f"Saved 3 figures to {fig_dir}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else None)

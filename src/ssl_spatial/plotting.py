"""Shared plotting style for the comparative-methods study
(`notebooks/spatial_methods_comparison.ipynb`), following the validated
categorical palette (CVD-safe fixed hue order) used elsewhere in this
project, extended to cover 10 methods without exceeding the ~8-series limit
on any single chart: methods are grouped into three families -- baselines
(including §2.6's re-weighted self-training), modern consistency-based SSL
(§2.4), and spatially explicit models (§2.4/H3, including the Fouedjio
geostatistical method) -- and every figure facets or filters by family
rather than drawing all 10 lines on one axis.

Titles are kept short (a noun phrase, not a full descriptive sentence); the
descriptive context that used to live in the in-plot title belongs in the
LaTeX figure caption instead, matching how figures are captioned in the
Fouedjio/Talebi paper this study extends (`Papers/`).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE_COLOR = "#c3c2b7"

# Validated 8-slot categorical palette (light mode), fixed order.
PALETTE = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
           "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]

METHOD_FAMILIES = {
    "baselines": ["supervised_only", "self_training", "label_propagation", "reweighted_self_training"],
    "modern_ssl": ["self_training", "mean_teacher", "fixmatch"],
    "spatial": ["supervised_only", "gcn", "spatial_gnn", "gnnwr", "geostatistical_ssl"],
}

METHOD_LABELS = {
    "supervised_only": "Supervised-only",
    "self_training": "Self-training",
    "label_propagation": "Label propagation",
    "reweighted_self_training": "Reweighted self-training",
    "mean_teacher": "Mean Teacher",
    "fixmatch": "FixMatch",
    "gcn": "GCN",
    "spatial_gnn": "Spatial GNN",
    "gnnwr": "GNNWR",
    "geostatistical_ssl": "Geostatistical SSL (Fouedjio)",
}

# One fixed colour per method, assigned from the 8-slot palette in a stable
# order; methods that appear in more than one family keep the same colour
# everywhere (colour follows the entity, never the panel).
METHOD_COLORS = {
    "supervised_only": PALETTE[0],
    "self_training": PALETTE[1],
    "label_propagation": PALETTE[2],
    "reweighted_self_training": PALETTE[3],
    "mean_teacher": PALETTE[4],
    "fixmatch": PALETTE[5],
    "gcn": PALETTE[6],
    "spatial_gnn": PALETTE[7],
    "gnnwr": PALETTE[4],
    "geostatistical_ssl": PALETTE[5],
}

METHOD_MARKERS = {
    "supervised_only": "o",
    "self_training": "s",
    "label_propagation": "^",
    "reweighted_self_training": "D",
    "mean_teacher": "o",
    "fixmatch": "s",
    "gcn": "^",
    "spatial_gnn": "D",
    "gnnwr": "v",
    "geostatistical_ssl": "P",
}


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BASELINE_COLOR)
    ax.spines["bottom"].set_color(BASELINE_COLOR)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)


def new_figure(figsize=(7, 4.5), ncols: int = 1, nrows: int = 1, sharey: bool = False):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=SURFACE, sharey=sharey)
    for ax in (np.atleast_1d(axes)).ravel():
        style_axes(ax)
    return fig, axes


def plot_method_lines(ax, df, x_col: str, y_col: str, methods: list[str], legend: bool = True) -> None:
    """Plot one line per method in `methods` (a single family), using each
    method's fixed colour/marker. Call `style_axes(ax)` beforehand if not
    using `new_figure`.
    """
    for method in methods:
        g = df[df["method"] == method].groupby(x_col)[y_col].mean()
        ax.plot(g.index, g.values, marker=METHOD_MARKERS[method], markersize=5, linewidth=1.8,
                color=METHOD_COLORS[method], label=METHOD_LABELS[method], zorder=3)
    if legend:
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY)


def savefig(fig, path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")


__all__ = [
    "SURFACE", "INK_PRIMARY", "INK_SECONDARY", "INK_MUTED", "GRIDLINE", "BASELINE_COLOR",
    "PALETTE", "METHOD_FAMILIES", "METHOD_LABELS", "METHOD_COLORS", "METHOD_MARKERS",
    "style_axes", "new_figure", "plot_method_lines", "savefig",
]

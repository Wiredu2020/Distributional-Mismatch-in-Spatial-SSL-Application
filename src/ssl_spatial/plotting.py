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

Reviewer fix (colour/typography pass requested on Figures 6 and 9 of
`manuscript/main/main.tex`, generalised to every figure in `figures/`): the
background is pure white rather than the previous off-white `#fcfcfb`
(which some PDF viewers/exporters render with a warm cast), the categorical
palette is Okabe & Ito's (2002) CVD-safe set rather than the previous
hand-picked one (`scripts/check_cvd_palette.py` simulates protanopia,
deuteranopia, and tritanopia on every colour combination that actually
co-occurs on one axis via `METHOD_FAMILIES`, and confirms the only residual
close pair -- blue/bluish-green under the rare tritanopia -- is disambiguated
by `METHOD_MARKERS`' distinct marker shapes), and every text element is
sized for print rather than screen.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#ffffff"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#3a3a3a"
INK_MUTED = "#5c5c5c"
GRIDLINE = "#dddddd"
BASELINE_COLOR = "#9a9a9a"

# Print-oriented type scale shared by every figure (points).
FONT_TITLE = 15
FONT_LABEL = 13
FONT_TICK = 12
FONT_LEGEND = 11
FONT_PANEL_LABEL = 16

# Okabe & Ito (2002) CVD-safe 8-colour categorical palette, fixed order.
# Verified with scripts/check_cvd_palette.py against every method subset
# that is actually plotted together (see module docstring).
PALETTE = ["#0072B2", "#009E73", "#E69F00", "#D55E00",
           "#CC79A7", "#56B4E9", "#F0E442", "#000000"]

METHOD_FAMILIES = {
    "baselines": ["supervised_only", "self_training", "label_propagation", "reweighted_self_training"],
    "modern_ssl": ["self_training", "mean_teacher", "fixmatch"],
    "spatial": ["supervised_only", "gcn", "spatial_gnn", "gnnwr", "geostatistical_ssl"],
    "h4_ablation": ["self_training", "reweighted_self_training", "reweighted_spatial_self_training",
                    "reweighted_adaptive_self_training", "full_distribution_aware"],
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
    "reweighted_spatial_self_training": "+ spatial weighting",
    "reweighted_adaptive_self_training": "+ adaptive threshold",
    "full_distribution_aware": "Full framework (all three)",
    "domain_adversarial_ssl": "Domain-adversarial SSL",
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
    "reweighted_spatial_self_training": PALETTE[6],
    "reweighted_adaptive_self_training": PALETTE[0],
    "full_distribution_aware": PALETTE[5],
    "domain_adversarial_ssl": PALETTE[0],
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
    "reweighted_spatial_self_training": "*",
    "reweighted_adaptive_self_training": "h",
    "full_distribution_aware": "X",
    "domain_adversarial_ssl": "p",
}


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BASELINE_COLOR)
    ax.spines["bottom"].set_color(BASELINE_COLOR)
    ax.tick_params(colors=INK_MUTED, labelsize=FONT_TICK)
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
        ax.plot(g.index, g.values, marker=METHOD_MARKERS[method], markersize=6, linewidth=2.0,
                color=METHOD_COLORS[method], label=METHOD_LABELS[method], zorder=3)
    if legend:
        ax.legend(frameon=False, fontsize=FONT_LEGEND, labelcolor=INK_SECONDARY)


def panel_label(ax, letter: str) -> None:
    """Draw a bold "(a)"-style panel label at the top-left of `ax`, in axes
    coordinates so it sits consistently outside the plotted data regardless
    of that panel's data range. Used in place of prose like "(left)"/
    "(right)" so every multi-panel figure in the paper shares one caption
    convention (a), (b), (c), ...
    """
    ax.text(-0.14, 1.10, f"({letter})", transform=ax.transAxes, fontsize=FONT_PANEL_LABEL,
             fontweight="bold", color=INK_PRIMARY, va="top", ha="left")


def savefig(fig, path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")


__all__ = [
    "SURFACE", "INK_PRIMARY", "INK_SECONDARY", "INK_MUTED", "GRIDLINE", "BASELINE_COLOR",
    "FONT_TITLE", "FONT_LABEL", "FONT_TICK", "FONT_LEGEND", "FONT_PANEL_LABEL",
    "PALETTE", "METHOD_FAMILIES", "METHOD_LABELS", "METHOD_COLORS", "METHOD_MARKERS",
    "style_axes", "new_figure", "plot_method_lines", "panel_label", "savefig",
]

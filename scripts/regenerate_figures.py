"""Regenerates every figure referenced by manuscript/main/{main,supplementary}.tex
with the reviewer-requested styling pass: pure-white background, a CVD-safe
palette (`ssl_spatial.plotting.PALETTE`), print-sized fonts, and a shared
(a)/(b)/(c) panel-label convention on every multi-panel figure in place of
prose like "(left)"/"(right)"/"(centre)".

Reads exclusively from the pre-computed `results/*.csv` files (and, for
Figure 7, directly from the small real-world source datasets) so that no
experiment is re-run -- several of the source notebook cells call
`*.run(...)`, which retrains models from scratch and would take far longer
than a styling fix warrants.

Usage:
    python scripts/regenerate_figures.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ssl_spatial.data.air_quality import load_air_quality
from ssl_spatial.data.housing import load_housing
from ssl_spatial.data.socioeconomic import load_socioeconomic
from ssl_spatial.metrics.bootstrap import bootstrap_ci_drop
from ssl_spatial.plotting import (
    BASELINE_COLOR, FONT_LABEL, FONT_LEGEND, FONT_TICK, FONT_TITLE,
    INK_PRIMARY, INK_SECONDARY, METHOD_COLORS, METHOD_FAMILIES, METHOD_LABELS, METHOD_MARKERS,
    PALETTE, new_figure, panel_label, plot_method_lines, savefig,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
FIG_DIR = REPO_ROOT / "figures"

BASE_METHODS = ["supervised_only", "self_training", "label_propagation"]
REAL_METHODS = ["supervised_only", "self_training", "label_propagation", "reweighted_self_training"]
DATASET_ORDER = ["synthetic", "wilds", "housing", "socioeconomic", "air_quality"]
DATASET_LABELS = {"synthetic": "Synthetic", "wilds": "PovertyMap-WILDS", "housing": "Housing",
                   "socioeconomic": "Socio-economic", "air_quality": "Air quality"}
ABLATION_ORDER = ["self_training", "reweighted_self_training", "reweighted_spatial_self_training",
                   "reweighted_adaptive_self_training", "full_distribution_aware"]
PANEL_LETTERS = "abcdefgh"


def fig1_accuracy_vs_mismatch() -> None:
    # Local override: label_propagation is red here only, not the shared PALETTE orange.
    colors = {**METHOD_COLORS, "label_propagation": "#D7191C"}

    synth = pd.read_csv(RESULTS / "controlled_experiment_v2.csv")
    stat_levels = sorted(synth["nonstationarity_strength"].unique())
    for letter, stat in zip(PANEL_LETTERS, stat_levels):
        sub = synth[synth["nonstationarity_strength"] == stat]
        fig, ax = new_figure(figsize=(6, 4.8))
        for method in BASE_METHODS:
            g = sub[sub["method"] == method].groupby("mismatch_alpha")["accuracy_out_region"].mean()
            ax.plot(g.index, g.values, marker=METHOD_MARKERS[method], markersize=7, linewidth=2,
                    color=colors[method], label=METHOD_LABELS[method], zorder=3)
        title = "Stationary label process" if stat == 0 else f"Non-stationary (strength={stat})"
        ax.set_title(title, fontsize=FONT_TITLE, color=INK_PRIMARY, loc="left")
        ax.set_xlabel(r"Mismatch level ($\alpha$)", fontsize=FONT_LABEL, color=INK_SECONDARY)
        ax.set_ylabel("Out-of-region accuracy", fontsize=FONT_LABEL, color=INK_SECONDARY)
        ax.legend(frameon=False, fontsize=FONT_LEGEND, loc="lower left", labelcolor=INK_SECONDARY)
        savefig(fig, FIG_DIR / f"fig1_accuracy_vs_mismatch_{letter}.png")


def fig2_spatial_gap_vs_mismatch() -> None:
    # Local override: label_propagation is red here only, not the shared PALETTE orange.
    colors = {**METHOD_COLORS, "label_propagation": "#D7191C"}

    synth = pd.read_csv(RESULTS / "controlled_experiment_v2.csv")
    fig, ax = new_figure(figsize=(7, 4.5))
    for method in BASE_METHODS:
        g = synth[synth["method"] == method].groupby("mismatch_alpha")["spatial_generalisation_gap"].mean()
        ax.plot(g.index, g.values, marker=METHOD_MARKERS[method], markersize=7, linewidth=2,
                color=colors[method], label=METHOD_LABELS[method], zorder=3)
    ax.axhline(0, color=BASELINE_COLOR, linewidth=1, zorder=1)
    ax.set_xlabel(r"Mismatch level ($\alpha$)", fontsize=FONT_LABEL, color=INK_SECONDARY)
    ax.set_ylabel("Spatial generalisation gap", fontsize=FONT_LABEL, color=INK_SECONDARY)
    ax.legend(frameon=False, fontsize=FONT_LEGEND, loc="upper left", labelcolor=INK_SECONDARY)
    savefig(fig, FIG_DIR / "fig2_spatial_gap_vs_mismatch.png")


def fig3_divergence_vs_accuracy() -> None:
    # Local override: label_propagation is red here only, not the shared PALETTE orange.
    colors = {**METHOD_COLORS, "label_propagation": "#D7191C"}

    synth = pd.read_csv(RESULTS / "controlled_experiment_v2.csv")
    fig, ax = new_figure(figsize=(7, 4.5))
    for method in BASE_METHODS:
        sub = synth[synth["method"] == method]
        ax.scatter(sub["mmd"], sub["accuracy_out_region"], s=26, alpha=0.6,
                   color=colors[method], label=METHOD_LABELS[method], zorder=3, edgecolors="none")
    ax.set_xlabel("Measured mismatch (MMD)", fontsize=FONT_LABEL, color=INK_SECONDARY)
    ax.set_ylabel("Out-of-region accuracy", fontsize=FONT_LABEL, color=INK_SECONDARY)
    ax.legend(frameon=False, fontsize=FONT_LEGEND, loc="lower left", labelcolor=INK_SECONDARY)
    savefig(fig, FIG_DIR / "fig3_divergence_vs_accuracy.png")


def fig4_5_wilds() -> None:
    # Local override: label_propagation is red here only, not the shared PALETTE orange.
    colors = {**METHOD_COLORS, "label_propagation": "#D7191C"}

    wilds = pd.read_csv(RESULTS / "wilds_povertymap_experiment_v2.csv")

    fig, ax = new_figure(figsize=(7, 4.5))
    for method in BASE_METHODS:
        g = wilds[wilds["method"] == method].groupby("mismatch_alpha")["accuracy_out_region"].mean()
        ax.plot(g.index, g.values, marker=METHOD_MARKERS[method], markersize=7, linewidth=2,
                color=colors[method], label=METHOD_LABELS[method], zorder=3)
    ax.legend(frameon=False, fontsize=FONT_LEGEND, loc="lower left", labelcolor=INK_SECONDARY)
    ax.set_xlabel("Mismatch severity (restricted source countries)", fontsize=FONT_LABEL, color=INK_SECONDARY)
    ax.set_ylabel("Out-of-region (OOD country) accuracy", fontsize=FONT_LABEL, color=INK_SECONDARY)
    savefig(fig, FIG_DIR / "fig4_wilds_accuracy_vs_mismatch.png")

    fig, ax = new_figure(figsize=(7, 4.5))
    for method in BASE_METHODS:
        g = wilds[wilds["method"] == method].groupby("mismatch_alpha")["spatial_generalisation_gap"].mean()
        ax.plot(g.index, g.values, marker=METHOD_MARKERS[method], markersize=7, linewidth=2,
                color=colors[method], label=METHOD_LABELS[method], zorder=3)
    ax.legend(frameon=False, fontsize=FONT_LEGEND, loc="upper left", labelcolor=INK_SECONDARY)
    ax.axhline(0, color=BASELINE_COLOR, linewidth=1, zorder=1)
    ax.set_xlabel("Mismatch severity (restricted source countries)", fontsize=FONT_LABEL, color=INK_SECONDARY)
    ax.set_ylabel("Spatial generalisation gap", fontsize=FONT_LABEL, color=INK_SECONDARY)
    savefig(fig, FIG_DIR / "fig5_wilds_spatial_gap_vs_mismatch.png")


def fig6_wilds_sample_patches() -> None:
    from ssl_spatial.data.wilds_poverty import download_metadata

    data_dir = REPO_ROOT / "data" / "wilds_povertymap"
    metadata = download_metadata(data_dir)
    cached_idx = sorted(int(p.stem.split("_")[-1]) for p in (data_dir / "images").glob("*.npz"))
    sample_meta = metadata.loc[metadata.index.isin(cached_idx)].copy()
    n_examples = 8
    picks = sample_meta.sort_values("wealthpooled").iloc[
        np.linspace(0, len(sample_meta) - 1, n_examples).astype(int)]

    band_idx = {"BLUE": 0, "GREEN": 1, "RED": 2}

    def rgb_composite(idx):
        img = np.load(data_dir / "images" / f"landsat_poverty_img_{idx}.npz")["x"]
        rgb = np.stack([img[band_idx["RED"]], img[band_idx["GREEN"]], img[band_idx["BLUE"]]], axis=-1)
        lo, hi = np.percentile(rgb, [2, 98])
        return np.clip((rgb - lo) / (hi - lo + 1e-9), 0, 1)

    fig, axes = new_figure(figsize=(14, 7.5), ncols=4, nrows=2)
    for ax, (idx, row) in zip(axes.ravel(), picks.iterrows()):
        ax.set_facecolor("white")
        ax.imshow(rgb_composite(idx))
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        region = "urban" if row["urban"] else "rural"
        ax.set_title(f"{row['country'].replace('_', ' ').title()} ({region})\n"
                     f"wealth index = {row['wealthpooled']:.2f}", fontsize=FONT_LABEL - 1, color=INK_PRIMARY)
    fig.subplots_adjust(hspace=0.5, wspace=0.15)
    savefig(fig, FIG_DIR / "fig6_wilds_sample_patches.png")


def fig7_realworld_datasets_overview() -> None:
    housing_df = load_housing(REPO_ROOT / "data" / "real_world" / "housing")
    socio_df = load_socioeconomic(REPO_ROOT / "data" / "real_world" / "socioeconomic")
    air_df = load_air_quality(REPO_ROOT / "data" / "real_world" / "air_quality",
                               REPO_ROOT / "data" / "real_world" / "socioeconomic")
    specs = [
        (housing_df, "California Housing", "a"), (socio_df, "US County Poverty", "b"),
        (air_df, "EPA PM2.5 Monitors", "c"),
    ]
    for df, title, letter in specs:
        fig, ax = new_figure(figsize=(6, 5.5))
        ax.scatter(df["lon"], df["lat"], c=df["label"], cmap="coolwarm", s=12, alpha=0.75, edgecolors="none")
        ax.set_title(title, fontsize=FONT_TITLE, color=INK_PRIMARY)
        ax.set_xlabel("Longitude", fontsize=FONT_LABEL, color=INK_SECONDARY)
        ax.set_ylabel("Latitude", fontsize=FONT_LABEL, color=INK_SECONDARY)
        ax.set_aspect("equal", adjustable="datalim")
        savefig(fig, FIG_DIR / f"fig7_realworld_datasets_overview_{letter}.png")


def fig8_realworld_accuracy_vs_mismatch() -> None:
    # Local override: label_propagation is red here only, not the shared PALETTE orange.
    colors = {**METHOD_COLORS, "label_propagation": "#D7191C"}

    titles = {"housing": "Housing (CA counties)", "socioeconomic": "Socio-economic (US states)",
              "air_quality": "Air quality (US states)"}
    for letter, name in zip(PANEL_LETTERS, ["housing", "socioeconomic", "air_quality"]):
        df = pd.read_csv(RESULTS / f"{name}_experiment_v2.csv")
        fig, ax = new_figure(figsize=(6, 4.8))
        for method in REAL_METHODS:
            g = df[df["method"] == method].groupby("mismatch_alpha")["accuracy_out_region"].mean()
            ax.plot(g.index, g.values, marker=METHOD_MARKERS[method], markersize=7, linewidth=2,
                    color=colors[method], label=METHOD_LABELS[method], zorder=3)
        ax.legend(frameon=False, fontsize=FONT_LEGEND, labelcolor=INK_SECONDARY)
        ax.set_title(titles[name], fontsize=FONT_TITLE, color=INK_PRIMARY)
        ax.set_xlabel(r"Mismatch level ($\alpha$)", fontsize=FONT_LABEL, color=INK_SECONDARY)
        ax.set_ylabel("Out-of-region accuracy", fontsize=FONT_LABEL, color=INK_SECONDARY)
        savefig(fig, FIG_DIR / f"fig8_realworld_accuracy_vs_mismatch_{letter}.png")


def fig9_h4_reweighting_comparison() -> None:
    rows = []
    synthetic_h4 = pd.read_csv(RESULTS / "controlled_experiment_h4.csv")
    wilds_h4 = pd.read_csv(RESULTS / "wilds_povertymap_h4.csv")

    def _drop_row(dataset, df):
        piv = df[df.method.isin(["self_training", "reweighted_self_training"])].groupby(
            ["method", "mismatch_alpha"])["accuracy_out_region"].mean().unstack()
        return {
            "dataset": dataset,
            "plain_drop_pp": (piv.loc["self_training", 0.0] - piv.loc["self_training", 1.0]) * 100,
            "reweighted_drop_pp": (piv.loc["reweighted_self_training", 0.0]
                                    - piv.loc["reweighted_self_training", 1.0]) * 100,
        }

    rows.append(_drop_row("synthetic", synthetic_h4))
    rows.append(_drop_row("wilds_povertymap", wilds_h4))
    for name in ["housing", "socioeconomic", "air_quality"]:
        rows.append(_drop_row(name, pd.read_csv(RESULTS / f"{name}_experiment_v2.csv")))
    h4_table = pd.DataFrame(rows)

    fig, ax = new_figure(figsize=(8.5, 5))
    x = np.arange(len(h4_table))
    width = 0.35
    ax.bar(x - width / 2, h4_table["plain_drop_pp"], width, label="Self-training (plain)",
           color=METHOD_COLORS["self_training"])
    ax.bar(x + width / 2, h4_table["reweighted_drop_pp"], width, label="Self-training (reweighted, §2.6)",
           color=METHOD_COLORS["reweighted_self_training"])
    ax.set_xticks(x)
    ax.set_xticklabels(h4_table["dataset"], rotation=20, ha="right", fontsize=FONT_TICK)
    ax.axhline(0, color=BASELINE_COLOR, linewidth=1)
    ax.set_ylabel("Accuracy drop, alpha=0 to alpha=1 (pp)", fontsize=FONT_LABEL, color=INK_SECONDARY)
    ax.legend(frameon=False, fontsize=FONT_LEGEND, labelcolor=INK_SECONDARY)
    savefig(fig, FIG_DIR / "fig9_h4_reweighting_comparison.png")


def fig16_h4_ablation_v2() -> None:
    # Local, figure-specific type scale (larger than the shared FONT_* constants):
    # this chart packs 5 dataset groups x 5 series onto one axis, so the shared
    # defaults read too small at print size. Bumped here only, not in
    # `ssl_spatial.plotting`, so no other figure's typography changes.
    tick_fs, label_fs, legend_fs = FONT_TICK + 4, FONT_LABEL + 4, FONT_LEGEND + 3

    drop = pd.read_csv(RESULTS / "h4_ablation_v2_drop_ci.csv")
    fig, ax = new_figure(figsize=(12.5, 7.2))
    x = np.arange(len(DATASET_ORDER))
    width = 0.16
    for i, method in enumerate(ABLATION_ORDER):
        sub = drop[drop["method"] == method].set_index("dataset").loc[DATASET_ORDER]
        offset = (i - (len(ABLATION_ORDER) - 1) / 2) * width
        ax.bar(x + offset, sub["drop_pp"], width, color=METHOD_COLORS[method], label=METHOD_LABELS[method],
               yerr=[sub["drop_pp"] - sub["ci_low_pp"], sub["ci_high_pp"] - sub["drop_pp"]],
               capsize=3, error_kw={"linewidth": 1.1, "ecolor": INK_SECONDARY})
    ax.axhline(0, color=BASELINE_COLOR, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASET_ORDER], fontsize=tick_fs)
    ax.tick_params(axis="y", labelsize=tick_fs)
    ax.set_ylabel("Accuracy drop, alpha=0 to alpha=1 (pp)", fontsize=label_fs, color=INK_SECONDARY)
    ax.set_ylim(top=ax.get_ylim()[1] + 6)  # headroom so the legend clears the tallest bar/error bar
    ax.legend(frameon=False, fontsize=legend_fs, labelcolor=INK_SECONDARY, ncol=2, loc="upper left")
    savefig(fig, FIG_DIR / "fig16_h4_ablation_v2.png")


def fig10_11_12_all_datasets() -> None:
    # Local, figure-specific type scale, larger than the shared FONT_* constants
    # (and no (a)/(b)/(c)/... panel letters on this trio, per request) -- bumped
    # here only, not in `ssl_spatial.plotting`, so no other figure changes.
    title_fs, label_fs, tick_fs, legend_fs = FONT_TITLE + 3, FONT_LABEL + 3, FONT_TICK + 3, FONT_LEGEND + 3

    df = pd.read_csv(RESULTS / "spatial_methods_comparison_v2.csv")
    specs = [
        ("baselines", "fig10_baselines_all_datasets.png"),
        ("modern_ssl", "fig11_modern_ssl_all_datasets.png"),
        ("spatial", "fig12_spatial_models_all_datasets.png"),
    ]
    for family, fname in specs:
        fig, axes = new_figure(figsize=(21, 4.8), ncols=5, sharey=False)
        for ax, dataset in zip(axes, DATASET_ORDER):
            sub = df[df["dataset"] == dataset]
            legend = dataset == "synthetic"
            plot_method_lines(ax, sub, "mismatch_alpha", "accuracy_out_region", METHOD_FAMILIES[family],
                               legend=legend)
            if legend:
                ax.legend(frameon=False, fontsize=legend_fs, labelcolor=INK_SECONDARY)
            ax.set_title(DATASET_LABELS[dataset], fontsize=title_fs, color=INK_PRIMARY)
            ax.set_xlabel("Mismatch severity", fontsize=label_fs, color=INK_SECONDARY)
            ax.tick_params(labelsize=tick_fs)
        axes[0].set_ylabel("Out-of-region accuracy", fontsize=label_fs, color=INK_SECONDARY)
        savefig(fig, FIG_DIR / fname)


def fig13_h3_spatial_drop_comparison() -> None:
    # Local, figure-specific type scale, larger than the shared FONT_* constants.
    # Bumped here only, not in `ssl_spatial.plotting`, so no other figure changes.
    tick_fs, label_fs, legend_fs = FONT_TICK + 4, FONT_LABEL + 4, FONT_LEGEND + 3

    # Local colour override matching two of fig13's five series to unused fig17
    # colours (GCN -> teal-green, Spatial GNN -> vermillion), so only GNNWR keeps
    # a colour absent from fig17 (no 4th fig17 colour left to give it). Kept
    # local to this function, like fig16's earlier override: `gcn`/`spatial_gnn`
    # are shared via METHOD_FAMILIES["spatial"] with fig12, which keeps its own
    # (different, previously-requested) yellow/black for those same two methods.
    spatial_drop_colors = {**METHOD_COLORS, "gcn": "#009E73", "spatial_gnn": "#D55E00"}

    df = pd.read_csv(RESULTS / "spatial_methods_comparison_v2.csv")
    drop = bootstrap_ci_drop(df, group_cols=["dataset", "method"], value_col="accuracy_out_region",
                              alpha_col="mismatch_alpha", n_boot=200, seed=0)
    drop["drop_pp"] = drop["drop"] * 100
    drop["ci_low_pp"] = drop["ci_low"] * 100
    drop["ci_high_pp"] = drop["ci_high"] * 100

    methods = METHOD_FAMILIES["spatial"]
    fig, ax = new_figure(figsize=(13, 6.5))
    x = np.arange(len(DATASET_ORDER))
    width = 0.8 / len(methods)
    for i, method in enumerate(methods):
        sub = drop[drop["method"] == method].set_index("dataset").reindex(DATASET_ORDER)
        offset = (i - (len(methods) - 1) / 2) * width
        ax.bar(x + offset, sub["drop_pp"], width, color=spatial_drop_colors[method], label=METHOD_LABELS[method],
               yerr=[sub["drop_pp"] - sub["ci_low_pp"], sub["ci_high_pp"] - sub["drop_pp"]],
               capsize=3, error_kw={"linewidth": 1.1, "ecolor": INK_SECONDARY})
    ax.axhline(0, color=BASELINE_COLOR, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASET_ORDER], fontsize=tick_fs)
    ax.tick_params(axis="y", labelsize=tick_fs)
    ax.set_ylabel("Accuracy drop, alpha=0 to alpha=1 (pp)", fontsize=label_fs, color=INK_SECONDARY)
    ax.set_ylim(top=ax.get_ylim()[1] + 6)
    ax.legend(frameon=False, fontsize=legend_fs, labelcolor=INK_SECONDARY, ncol=2, loc="upper left")
    savefig(fig, FIG_DIR / "fig13_h3_spatial_drop_comparison.png")


def fig14_changepoint_analysis() -> None:
    import pwlf

    # Local override, line and breakpoint both: self-training red, label propagation green.
    colors = {**METHOD_COLORS, "self_training": "#D7191C", "label_propagation": "#009E73"}
    # Same blue for supervised-only's line and breakpoint (no separate deep-blue override).
    breakpoint_colors = colors

    sens_df = pd.read_csv(RESULTS / "controlled_experiment_sensitivity.csv")
    breakpoints = pd.read_csv(RESULTS / "changepoint_analysis.csv")
    metric_titles = {"accuracy_out_region": "Out-of-region accuracy",
                      "spatial_generalisation_gap": "Spatial generalisation gap"}

    for letter, metric in zip(PANEL_LETTERS, ["accuracy_out_region", "spatial_generalisation_gap"]):
        fig, ax = new_figure(figsize=(6, 4.8))
        for method in BASE_METHODS:
            sub = sens_df[sens_df["method"] == method]
            curve = sub.groupby("mismatch_alpha")[metric].mean().reset_index()
            xv, yv = curve["mismatch_alpha"].to_numpy(), curve[metric].to_numpy()
            ax.plot(xv, yv, marker=METHOD_MARKERS[method], markersize=7, linewidth=0, color=colors[method],
                    label=METHOD_LABELS[method], zorder=4)
            model = pwlf.PiecewiseLinFit(xv, yv, seed=0)
            model.fit(2)
            x_fit = np.linspace(xv.min(), xv.max(), 100)
            ax.plot(x_fit, model.predict(x_fit), linestyle="-", linewidth=1.8, color=colors[method],
                    alpha=0.8, zorder=3)
            bp_row = breakpoints[(breakpoints["method"] == method) & (breakpoints["metric"] == metric)]
            if len(bp_row):
                ax.axvline(bp_row.iloc[0]["breakpoint"], color=breakpoint_colors[method], linestyle="--",
                           linewidth=1.3, alpha=0.7, zorder=2)
        ax.set_title(metric_titles[metric], fontsize=FONT_TITLE, color=INK_PRIMARY)
        ax.set_xlabel(r"Mismatch level ($\alpha$)", fontsize=FONT_LABEL, color=INK_SECONDARY)
        ax.set_ylabel("Value", fontsize=FONT_LABEL, color=INK_SECONDARY)
        ax.legend(frameon=False, fontsize=FONT_LEGEND, labelcolor=INK_SECONDARY, loc="best")
        savefig(fig, FIG_DIR / f"fig14_changepoint_analysis_{letter}.png")


def fig15_kernel_weighted_mmd_validation() -> None:
    corr = pd.read_csv(RESULTS / "localized_mmd_comparison_v2_correlation_ci.csv")
    metric_short = {"kl_divergence": "KL divergence", "mmd": "MMD (RBF)", "wasserstein": "Wasserstein",
                     "localized_mmd": "Localised MMD\n(fixed grid)",
                     "kernel_weighted_local_mmd": "Localised MMD\n(kernel-weighted)"}
    corr_plot = corr.copy()
    corr_plot["label"] = corr_plot["metric"].map(metric_short)

    fig, ax = new_figure(figsize=(7.5, 4.8))
    x = np.arange(len(corr_plot))
    ax.bar(x, corr_plot["r"], color=PALETTE[0],
           yerr=[corr_plot["r"] - corr_plot["ci_low"], corr_plot["ci_high"] - corr_plot["r"]],
           capsize=3, error_kw={"linewidth": 1, "ecolor": INK_SECONDARY})
    ax.axhline(0, color=BASELINE_COLOR, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(corr_plot["label"], fontsize=FONT_TICK - 1)
    ax.set_ylabel("Pearson r with out-of-region accuracy", fontsize=FONT_LABEL, color=INK_SECONDARY)
    savefig(fig, FIG_DIR / "fig15_kernel_weighted_mmd_validation.png")


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    steps = [
        fig1_accuracy_vs_mismatch, fig2_spatial_gap_vs_mismatch, fig3_divergence_vs_accuracy,
        fig4_5_wilds, fig6_wilds_sample_patches, fig7_realworld_datasets_overview,
        fig8_realworld_accuracy_vs_mismatch, fig9_h4_reweighting_comparison, fig16_h4_ablation_v2,
        fig10_11_12_all_datasets, fig13_h3_spatial_drop_comparison, fig14_changepoint_analysis,
        fig15_kernel_weighted_mmd_validation,
    ]
    for step in steps:
        print(f"-- {step.__name__}")
        step()
    print("done")


if __name__ == "__main__":
    main()

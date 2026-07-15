"""Formal changepoint / sensitivity analysis (manuscript methods.tex
Section 2.5, Step 3 ToDo): fits a two-segment piecewise-linear
(segmented-regression) model to out-of-region accuracy and the spatial
generalisation gap as a function of mismatch severity, separately per
method, and reports the estimated breakpoint location with a bootstrap
confidence interval, replacing the visual "alpha ~ 0.6" read from the
figures with a fitted estimate.

Requires the higher-seed-count re-run in
`results/controlled_experiment_sensitivity.csv` (20 seeds instead of 3;
see `configs/controlled_experiment_sensitivity.yaml`), since three seeds
were judged too few to support a precise interval.

Usage:
    python -m ssl_spatial.experiments.changepoint_analysis
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pwlf

REPO_ROOT = Path(__file__).resolve().parents[3]


def _fit_breakpoint(alpha: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
    model = pwlf.PiecewiseLinFit(alpha, y, seed=seed)
    breaks = model.fit(2)  # 2 segments -> 1 interior breakpoint
    return float(breaks[1])


def _bootstrap_breakpoint_ci(df_method: pd.DataFrame, metric: str, n_boot: int = 200,
                              seed: int = 0) -> tuple[float, float, float]:
    """Point estimate from the full (seed-averaged) curve, plus a
    percentile bootstrap CI by resampling seeds with replacement.
    """
    seeds = df_method["seed"].unique()
    rng = np.random.default_rng(seed)

    point_curve = df_method.groupby("mismatch_alpha")[metric].mean().reset_index()
    point_estimate = _fit_breakpoint(point_curve["mismatch_alpha"].to_numpy(), point_curve[metric].to_numpy())

    boot_breaks = []
    for b in range(n_boot):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        boot_df = pd.concat([df_method[df_method["seed"] == s] for s in sampled_seeds], ignore_index=True)
        curve = boot_df.groupby("mismatch_alpha")[metric].mean().reset_index()
        try:
            brk = _fit_breakpoint(curve["mismatch_alpha"].to_numpy(), curve[metric].to_numpy(), seed=b)
            boot_breaks.append(brk)
        except Exception:
            continue

    boot_breaks = np.array(boot_breaks)
    lo, hi = np.percentile(boot_breaks, [2.5, 97.5])
    return point_estimate, float(lo), float(hi)


def run(results_csv: str | None = None, n_boot: int = 200) -> pd.DataFrame:
    results_csv = results_csv or str(REPO_ROOT / "results" / "controlled_experiment_sensitivity.csv")
    df = pd.read_csv(results_csv)

    rows = []
    for method in sorted(df["method"].unique()):
        df_m = df[df["method"] == method]
        for metric in ["accuracy_out_region", "spatial_generalisation_gap"]:
            point, lo, hi = _bootstrap_breakpoint_ci(df_m, metric, n_boot=n_boot)
            rows.append({"method": method, "metric": metric, "breakpoint": point,
                         "ci_low": lo, "ci_high": hi})
            print(f"{method:20s} {metric:28s} breakpoint={point:.3f}  95% CI=[{lo:.3f}, {hi:.3f}]",
                  file=sys.stderr)

    out = pd.DataFrame(rows)
    out_path = REPO_ROOT / "results" / "changepoint_analysis.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved {len(out)} rows to {out_path}", file=sys.stderr)
    return out


if __name__ == "__main__":
    run()

"""Shared bootstrap confidence-interval utilities (statistical-rigor fix
requested in the supervisor review): generalises the seed-cluster percentile
bootstrap already used for the changepoint breakpoint CI
(`ssl_spatial.experiments.changepoint_analysis._bootstrap_breakpoint_ci`) so
accuracy/drop tables and the divergence-vs-accuracy correlation can carry the
same honest CIs, instead of being reported as bare point estimates.

The resampling unit throughout is the *seed*, not the row: within one seed,
every (alpha, lengthscale, nonstationarity, method) combination is already
fully crossed by design, so seeds are the actual independent replicates, and
resampling individual rows would understate uncertainty -- the pseudo-
replication the supervisor review flagged in the pooled correlation
(`ssl_spatial.experiments.localized_mmd_analysis`). Every function here
resamples whole seeds with replacement, recomputes the statistic of interest
on the resampled data, and reports a percentile interval, matching
`changepoint_analysis.py`'s existing convention (`n_boot=200`,
`np.random.default_rng`, 95% = `[2.5, 97.5]` percentiles) rather than
introducing a different methodology.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd


def _seed_resamples(df: pd.DataFrame, seed_col: str, n_boot: int, seed: int) -> Iterator[pd.DataFrame]:
    """Yield `n_boot` resampled copies of `df`, each built by sampling the
    unique values of `seed_col` with replacement and concatenating the
    corresponding row blocks -- the same resampling scheme as
    `changepoint_analysis._bootstrap_breakpoint_ci`.
    """
    seeds = df[seed_col].unique()
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        sampled = rng.choice(seeds, size=len(seeds), replace=True)
        yield pd.concat([df[df[seed_col] == s] for s in sampled], ignore_index=True)


def bootstrap_ci_mean(df: pd.DataFrame, group_cols: list[str], value_col: str,
                       seed_col: str = "seed", n_boot: int = 200, seed: int = 0) -> pd.DataFrame:
    """95% seed-cluster bootstrap CI on the mean of `value_col` within each
    `group_cols` group (e.g. accuracy at a given method/mismatch_alpha).
    Returns one row per group with columns [*group_cols, value_col, ci_low, ci_high].
    """
    point = df.groupby(group_cols)[value_col].mean()
    boot = [resampled.groupby(group_cols)[value_col].mean()
            for resampled in _seed_resamples(df, seed_col, n_boot, seed)]
    boot_df = pd.concat(boot, axis=1)
    out = point.to_frame(value_col)
    out["ci_low"] = boot_df.quantile(0.025, axis=1)
    out["ci_high"] = boot_df.quantile(0.975, axis=1)
    return out.reset_index()


def bootstrap_ci_drop(df: pd.DataFrame, group_cols: list[str], value_col: str, alpha_col: str,
                       seed_col: str = "seed", lo: float = 0.0, hi: float = 1.0,
                       n_boot: int = 200, seed: int = 0) -> pd.DataFrame:
    """95% seed-cluster bootstrap CI on the drop `value_col(alpha=lo) -
    value_col(alpha=hi)` within each `group_cols` group -- the "accuracy
    drop, alpha=0 to alpha=1" reported throughout the manuscript's tables.
    Returns one row per group with columns [*group_cols, drop, ci_low, ci_high].
    """
    def _drop(frame: pd.DataFrame) -> pd.Series:
        means = frame.groupby(group_cols + [alpha_col])[value_col].mean().unstack(alpha_col)
        return means[lo] - means[hi]

    point = _drop(df)
    boot = [_drop(resampled) for resampled in _seed_resamples(df, seed_col, n_boot, seed)]
    boot_df = pd.concat(boot, axis=1)
    out = point.to_frame("drop")
    out["ci_low"] = boot_df.quantile(0.025, axis=1)
    out["ci_high"] = boot_df.quantile(0.975, axis=1)
    return out.reset_index()


def bootstrap_ci_correlation(df: pd.DataFrame, x_col: str, y_col: str,
                              seed_col: str = "seed", n_boot: int = 200,
                              seed: int = 0, method: str = "pearson") -> tuple[float, float, float]:
    """95% seed-cluster bootstrap CI around the pooled correlation between
    `x_col` and `y_col` (e.g. a divergence metric vs out-of-region accuracy
    across all synthetic runs). `method` is passed straight to
    `DataFrame.corr` ("pearson" or "spearman"); the CI makes explicit how
    much of the apparent precision survives once the seed/lengthscale/
    nonstationarity nesting is accounted for, rather than treating all rows
    as independent observations.
    """
    point = float(df[[x_col, y_col]].corr(method=method).iloc[0, 1])
    boots = []
    for resampled in _seed_resamples(df, seed_col, n_boot, seed):
        r = resampled[[x_col, y_col]].corr(method=method).iloc[0, 1]
        if np.isfinite(r):
            boots.append(r)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


__all__ = ["bootstrap_ci_mean", "bootstrap_ci_drop", "bootstrap_ci_correlation"]

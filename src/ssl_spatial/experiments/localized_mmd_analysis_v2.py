"""Re-validates the localised-divergence estimator (manuscript methods.tex
Section 2.3), with the statistical-rigor fix requested in the supervisor
review: the original `localized_mmd_analysis.py` reports a single pooled
Pearson correlation over all 324 rows of `results/controlled_experiment.csv`,
pseudo-replicated because rows are nested by seed x lengthscale x
nonstationarity_strength x mismatch_alpha x method rather than independent,
which inflates the apparent precision of Table 2's headline numbers. This
version reuses the exact same divergence computation and merge logic, but
(a) runs on `configs/controlled_experiment_v2.yaml` / `results/
controlled_experiment_v2.csv` (20 seeds instead of 3), and (b) reports a 95%
seed-cluster bootstrap CI around each pooled correlation
(`ssl_spatial.metrics.bootstrap.bootstrap_ci_correlation`), so the reader can
see how much of that correlation's apparent precision survives once the
nesting is accounted for, rather than a bare point estimate.

Usage:
    python -m ssl_spatial.experiments.localized_mmd_analysis_v2
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pandas as pd
import yaml

from ssl_spatial.data.synthetic import SpatialSSLConfig, generate_spatial_ssl_dataset
from ssl_spatial.metrics.bootstrap import bootstrap_ci_correlation
from ssl_spatial.metrics.divergence import kernel_weighted_local_mmd

REPO_ROOT = Path(__file__).resolve().parents[3]

METRICS = ["kl_divergence", "wasserstein", "mmd", "localized_mmd", "kernel_weighted_local_mmd"]


def run(config_path: str | None = None, results_csv: str | None = None, n_boot: int = 200) -> pd.DataFrame:
    config_path = config_path or str(REPO_ROOT / "configs" / "controlled_experiment_v2.yaml")
    results_csv = results_csv or str(REPO_ROOT / "results" / "controlled_experiment_v2.csv")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    sweep, ds_cfg, div_cfg = cfg["sweep"], cfg["dataset"], cfg["divergence"]

    combos = list(itertools.product(
        sweep["mismatch_alpha"], sweep["lengthscale"], sweep["nonstationarity_strength"], sweep["seeds"],
    ))

    rows = []
    for i, (alpha, lengthscale, nonstat, seed) in enumerate(combos):
        data_cfg = SpatialSSLConfig(
            mismatch_alpha=alpha, lengthscale=lengthscale, nonstationarity_strength=nonstat,
            label_anchor=tuple(ds_cfg["label_anchor"]), bandwidth_min=ds_cfg["bandwidth_min"],
            bandwidth_max=ds_cfg["bandwidth_max"], grid_resolution=ds_cfg["grid_resolution"],
            n_features=ds_cfg["n_features"], n_labelled=ds_cfg["n_labelled"],
            n_unlabelled=ds_cfg["n_unlabelled"], n_test_in=ds_cfg["n_test_in"],
            n_test_out=ds_cfg["n_test_out"], noise_std=ds_cfg["noise_std"],
        )
        dataset = generate_spatial_ssl_dataset(data_cfg, seed=seed)
        X_l, X_u = dataset.X[dataset.idx_labelled], dataset.X[dataset.idx_unlabelled]
        c_l, c_u = dataset.coords[dataset.idx_labelled], dataset.coords[dataset.idx_unlabelled]

        result = kernel_weighted_local_mmd(c_l, X_l, c_u, X_u, n_grid=div_cfg["n_bins"])
        rows.append({
            "mismatch_alpha": alpha, "lengthscale": lengthscale,
            "nonstationarity_strength": nonstat, "seed": seed,
            "kernel_weighted_local_mmd": result["aggregate"],
        })

        if (i + 1) % 100 == 0 or (i + 1) == len(combos):
            print(f"[{i + 1}/{len(combos)}] combos done", file=sys.stderr)

    new_metric = pd.DataFrame(rows)
    existing = pd.read_csv(results_csv)
    merged = existing.merge(new_metric, on=["mismatch_alpha", "lengthscale", "nonstationarity_strength", "seed"])

    print("\nPooled Pearson correlation with out-of-region accuracy, 95% seed-cluster bootstrap CI "
          f"({n_boot} resamples) -- point estimate is the same naive pooled corr() as the original analysis; "
          "the CI shows how much precision survives once the seed/lengthscale/nonstationarity nesting is "
          "accounted for:", file=sys.stderr)
    ci_rows = []
    for metric in METRICS:
        point, lo, hi = bootstrap_ci_correlation(merged, metric, "accuracy_out_region", n_boot=n_boot)
        ci_rows.append({"metric": metric, "r": point, "ci_low": lo, "ci_high": hi})
        print(f"  {metric:28s} r={point:6.3f}   95% CI=[{lo:6.3f}, {hi:6.3f}]", file=sys.stderr)

    ci_table = pd.DataFrame(ci_rows)
    ci_out_path = REPO_ROOT / "results" / "localized_mmd_comparison_v2_correlation_ci.csv"
    ci_table.to_csv(ci_out_path, index=False)

    out_path = REPO_ROOT / "results" / "localized_mmd_comparison_v2.csv"
    merged.to_csv(out_path, index=False)
    print(f"\nSaved merged comparison to {out_path}", file=sys.stderr)
    print(f"Saved correlation CIs to {ci_out_path}", file=sys.stderr)
    return merged


if __name__ == "__main__":
    run()

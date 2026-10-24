"""Controlled unlabelled-pool ablation (reviewer-requested addition): fixes the
labelled set and both test sets at a realistic, spatially biased draw, and
varies only the unlabelled pool's own sampling distribution -- none, matched
to the target domain, partially biased towards the labelled region, or fully
concentrated there -- isolating whether a *mismatched unlabelled pool itself*
degrades performance, as opposed to reduced labelled coverage
(`controlled_experiment.py`'s mismatch_alpha sweep never varies the unlabelled
pool: it is always uniform there).

Usage:
    python -m ssl_spatial.experiments.unlabelled_mismatch_experiment [config.yaml]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import yaml

from ssl_spatial.data.synthetic import SpatialSSLConfig, generate_spatial_ssl_dataset
from ssl_spatial.metrics.evaluation import evaluate_classifier
from ssl_spatial.models.baselines import BASELINE_METHODS

REPO_ROOT = Path(__file__).resolve().parents[3]

# unlabelled_alpha=None used only for the "none" condition (n_unlabelled forced to 0 there
# regardless of the config value, so the field is irrelevant).
CONDITIONS = {
    "none": None,
    "matched": 0.0,
    "partial": 0.5,
    "strong": 1.0,
}


def run(config_path: str) -> pd.DataFrame:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    ds_cfg = cfg["dataset"]
    seeds = cfg["seeds"]
    methods = cfg["methods"]

    rows = []
    start = time.time()
    combos = list(seeds)
    for i, seed in enumerate(combos):
        for condition, unlabelled_alpha in CONDITIONS.items():
            n_unlabelled = 0 if condition == "none" else ds_cfg["n_unlabelled"]
            data_cfg = SpatialSSLConfig(
                mismatch_alpha=ds_cfg["mismatch_alpha"],
                lengthscale=ds_cfg["lengthscale"],
                nonstationarity_strength=ds_cfg["nonstationarity_strength"],
                label_anchor=tuple(ds_cfg["label_anchor"]),
                bandwidth_min=ds_cfg["bandwidth_min"],
                bandwidth_max=ds_cfg["bandwidth_max"],
                grid_resolution=ds_cfg["grid_resolution"],
                n_features=ds_cfg["n_features"],
                n_labelled=ds_cfg["n_labelled"],
                n_unlabelled=n_unlabelled,
                n_test_in=ds_cfg["n_test_in"],
                n_test_out=ds_cfg["n_test_out"],
                noise_std=ds_cfg["noise_std"],
                unlabelled_alpha=unlabelled_alpha,
            )
            dataset = generate_spatial_ssl_dataset(data_cfg, seed=seed)

            X_l, y_l = dataset.X[dataset.idx_labelled], dataset.y[dataset.idx_labelled]
            X_u = dataset.X[dataset.idx_unlabelled]
            X_in, y_in = dataset.X[dataset.idx_test_in], dataset.y[dataset.idx_test_in]
            X_out, y_out = dataset.X[dataset.idx_test_out], dataset.y[dataset.idx_test_out]

            method_list = ["supervised_only"] if condition == "none" else methods
            for method in method_list:
                fit_fn = BASELINE_METHODS[method]
                clf = fit_fn(X_l, y_l, X_u, seed=seed)
                perf = evaluate_classifier(clf, X_in, y_in, X_out, y_out)
                rows.append({
                    "seed": seed,
                    "condition": condition,
                    "method": method,
                    "n_unlabelled": n_unlabelled,
                    **perf,
                })

        if (i + 1) % 5 == 0 or (i + 1) == len(combos):
            elapsed = time.time() - start
            print(f"[{i + 1}/{len(combos)}] seeds done ({elapsed:.1f}s elapsed)", file=sys.stderr)

    results = pd.DataFrame(rows)
    out_path = REPO_ROOT / cfg["output_csv"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    print(f"Saved {len(results)} rows to {out_path}", file=sys.stderr)
    return results


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else str(
        REPO_ROOT / "configs" / "unlabelled_mismatch_experiment.yaml")
    run(config_path)

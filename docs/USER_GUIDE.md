# User guide and worked example

## Purpose and scope

This package evaluates semi-supervised classifiers as labelled geographic
coverage contracts. The synthetic example needs no external data and is the
recommended installation test. It evaluates a prescribed experiment; it does
not certify that a model is safe for deployment.

## Inputs and options

Run commands from the repository root after setting `PYTHONPATH` to `src`.
Experiment runners accept one positional YAML configuration path, not arbitrary
command-line flags. Copy a configuration before changing its options.

| Synthetic YAML option | Meaning |
| --- | --- |
| `output_csv` | Output CSV path, relative to the repository root |
| `sweep.mismatch_alpha` | List of label-sampling severity values (0 to 1) |
| `sweep.lengthscale` | Gaussian smoothing widths as fractions of domain width |
| `sweep.nonstationarity_strength` | Strength of spatial variation in the response coefficient |
| `sweep.seeds` | Random seeds used in every parameter combination |
| `methods` | Registered method names; smoke test uses supervised_only, self_training, label_propagation |
| `dataset.grid_resolution` | Number of grid cells per axis |
| `dataset.n_features` | Number of synthetic features |
| `dataset.n_labelled`, `n_unlabelled`, `n_test_in`, `n_test_out` | Sample sizes; their total must not exceed grid_resolution squared |
| `dataset.noise_std` | Standard deviation of independent latent-response noise |
| `dataset.label_anchor` | Two coordinates of the label-sampling centre |
| `dataset.bandwidth_min`, `bandwidth_max` | Sampling bandwidth endpoints |
| `divergence.n_bins`, `min_count` | Local divergence grid size and minimum observations per group |

Real-data configurations additionally specify local data directories, region
split settings and, for WILDS, country fold, candidate pool, PCA and download
settings. Use the configuration matching the experiment; YAML schemas differ
between runners. See DATA.md for required input tables and headers.

## Tutorial: run and inspect the synthetic test

```bash
python -m ssl_spatial.experiments.controlled_experiment configs/smoke_test.yaml
python -c "import pandas as pd; d=pd.read_csv('results/smoke_test.csv'); print(d[['method','mismatch_alpha','accuracy_out_region']].to_string(index=False))"
```

Expect nine rows: three methods at three severity levels. Rows identify the
seed, length-scale, non-stationarity strength, method, and severity. Accuracy,
F1 and calibration concern the specified held-out samples. The spatial gap is
in-region accuracy minus out-of-region accuracy; larger positive values mean
worse performance outside the labelled sampling footprint. Divergences compare
labelled and unlabelled feature samples.

A second run with identical inputs and software versions should reproduce the
same baseline outputs. The runner overwrites the configured output CSV. Copy
results before changing a configuration or rerunning an experiment.

## Tutorial: twenty-seed results

```bash
python -m ssl_spatial.experiments.controlled_experiment configs/controlled_experiment_v2.yaml
python -m ssl_spatial.experiments.localized_mmd_analysis_v2
```

The first command writes 2,160 fitted-model rows. The second consumes that CSV
and recomputes local-MMD diagnostics and bootstrap summaries. This is a larger
analysis and is not needed for installation verification. Consult
REPRODUCIBILITY.md for the remaining runners and output mapping.

## Expected behaviour and errors

Progress is written to the terminal and results to the configured CSV. Missing
source modules indicate an incomplete checkout; missing local data files require
data preparation. Insufficient grid size raises a sample-size error. Some model
fits can issue convergence warnings, which should be investigated rather than
silently discarded. Optional neural methods require PyTorch; geostatistical
methods require GSTools. The full study has not been rerun during this revision.

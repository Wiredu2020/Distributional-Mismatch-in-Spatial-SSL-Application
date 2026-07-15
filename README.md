# Marginal Distribution Mismatch in Spatial SSL — code companion

Code companion to `Marginal distribution mismatch study for SSL.docx`. Implements
the controlled-experiment phase of the methodology (§6.3–6.5): synthetic spatial
data with tunable mismatch/autocorrelation/non-stationarity, distribution
divergence metrics, and SSL baseline comparisons.

## Structure

```
src/ssl_spatial/
  data/synthetic.py           Synthetic spatial data generator (§6.2a)
  data/wilds_poverty.py       WILDS PovertyMap loader/feature engineering (§2.2.2 benchmark eval)
  data/region_mismatch.py     Generic region-restriction mismatch sweep (shared by the 3 real-world datasets)
  data/housing.py             California Housing loader (§2.2.3, housing domain)
  data/socioeconomic.py       USDA county poverty/income/education loader (§2.2.3, socio-economic domain)
  data/air_quality.py         EPA AQS PM2.5 monitor loader (§2.2.3, environmental domain)
  metrics/divergence.py       KL, Wasserstein, MMD + spatially-localised versions (§6.3)
  metrics/evaluation.py       Accuracy/F1, calibration error, spatial generalisation gap (§6.5 step 2)
  models/baselines.py         Supervised-only, self-training, label propagation (§6.4)
  models/distribution_aware.py  Density-ratio reweighted self-training (§2.6, H4 test)
  experiments/
    controlled_experiment.py  Synthetic sweep runner (§6.5 step 1)
    wilds_benchmark.py        WILDS PovertyMap sweep runner (§2.2.2 benchmark eval)
    real_world_benchmark.py   Generic runner for housing/socioeconomic/air_quality (§2.2.3)
    plot_results.py           Figures for H1/H2 evidence
notebooks/
  benchmark.ipynb              WILDS PovertyMap benchmark walkthrough (§2.2.2 ToDo)
  real_world_benchmark.ipynb   Real-world datasets + §2.6/H4 walkthrough (§2.2.3 ToDo)
configs/controlled_experiment.yaml       Synthetic sweep definition
configs/controlled_experiment_h4.yaml    Synthetic sweep + reweighted_self_training (H4 test)
configs/wilds_benchmark.yaml             WILDS PovertyMap sweep definition
configs/wilds_benchmark_h4.yaml          WILDS sweep + reweighted_self_training (H4 test)
configs/housing_benchmark.yaml           California Housing sweep definition
configs/socioeconomic_benchmark.yaml     USDA county sweep definition
configs/air_quality_benchmark.yaml       EPA AQS sweep definition
results/                                  One CSV per config above (gitignored)
data/wilds_povertymap/                    Downloaded WILDS PovertyMap metadata + image cache (gitignored)
data/real_world/                          Downloaded housing/socioeconomic/air_quality source files (gitignored)
figures/                                  fig1-3 (synthetic), fig4-6 (WILDS), fig7-9 (real-world + H4) PNGs
```

## How the synthetic generator maps to the manuscript's concepts

`SpatialSSLConfig` (`src/ssl_spatial/data/synthetic.py`) exposes the three axes
of violation the manuscript separates conceptually:

- `mismatch_alpha` — **marginal distribution mismatch**. Labelled data is sampled
  from a Gaussian kernel concentrated at `label_anchor`; unlabelled/test data is
  sampled uniformly over the domain. `alpha=0` → same distribution, `alpha=1` →
  labelled data confined to a small sub-region. This is a stand-in for real
  sampling bias (e.g. monitoring stations clustered in urban cores).
- `nonstationarity_strength` — **spatial non-stationarity**. The coefficient
  linking covariates to the label varies smoothly across space, independent of
  where data happens to be sampled. This is a *concept*-shift axis, distinct
  from the *covariate*-shift axis above — the manuscript's research questions
  treat these as separate mechanisms and the code keeps them separately
  controllable.
- `lengthscale` — **spatial autocorrelation** in the covariates themselves
  (smoothed random field bandwidth).

## Reproducing the results

```bash
pip install -r requirements.txt
PYTHONPATH=src python3 -m ssl_spatial.experiments.controlled_experiment configs/controlled_experiment.yaml
PYTHONPATH=src python3 -m ssl_spatial.experiments.plot_results
```

Full sweep (6 mismatch levels × 3 lengthscales × 2 non-stationarity levels ×
3 seeds × 3 methods = 324 runs) takes ~2 minutes on a laptop CPU.

## What this pilot already shows (see `results/findings_notes.md`)

Preliminary evidence for **H1** (performance degrades with mismatch, sharply
past α≈0.6) and partial/nuanced evidence for **H2** (self-training degrades
fastest in absolute terms; label propagation is comparatively more robust,
which complicates a simple "manifold methods are most fragile" reading — worth
digging into in the write-up). Full findings, caveats, and exact numbers are in
`results/findings_notes.md` — written so it can be lifted into the manuscript's
methodology/results text directly.

## Benchmark dataset evaluation: WILDS PovertyMap (§2.2.2)

`notebooks/benchmark.ipynb` re-runs the controlled-mismatch methodology on
**PovertyMap-WILDS** \citep{koh2021wilds}, which has an official country-level
geographic split. Mismatch severity is operationalised by restricting the
labelled set to a shrinking, nested subset of the fold's training countries
(mirroring the synthetic `mismatch_alpha` sweep), with unlabelled/in-region-test
data drawn uniformly across all training countries and out-of-region test data
drawn from the fold's official held-out OOD countries. Images are pulled
individually from CodaLab's file-level REST endpoint (rather than the ~13GB
full archive) and reduced to per-band summary-statistic features, PCA-reduced
to keep the KDE-based KL-divergence estimator well-posed. Reproduce with:

```bash
PYTHONPATH=src python3 -m ssl_spatial.experiments.wilds_benchmark configs/wilds_benchmark.yaml
```

(after downloading the pool's images once — see the notebook's Section 3 — since
the runner itself does no network I/O).

## Real-world datasets (§2.2.3) and the §2.6 distribution-aware framework

`notebooks/real_world_benchmark.ipynb` sources one real, tabular, minimal-size
dataset per motivating domain — California Housing (`sklearn`, auto-downloaded),
USDA county-level poverty/income/education (direct CSV/XLSX, no API key), and
EPA AQS PM2.5 monitors (direct pre-generated file, no API key) — and runs each
through the same region-restriction mismatch sweep used for WILDS
(`src/ssl_spatial/data/region_mismatch.py`), with county/state as the region
unit and the nested severity schedule anchored on the largest training region
(rather than alphabetical) so it stays spatially concentrated and large enough
to draw labelled samples from at every severity level.

The notebook also implements and tests the manuscript's §2.6 recommended
starting point for the distribution-aware framework: `reweighted_self_training`
(`src/ssl_spatial/models/distribution_aware.py`) reweights labelled points by
an estimated density ratio (logistic-regression discriminator odds) and layers
this on top of self-training. It's wired in as a 4th method alongside the three
baselines in every runner, so H4 (does reweighting mitigate the mismatch-induced
drop?) can be tested via `configs/*_h4.yaml` / the real-world configs directly.
Finding: reweighting meaningfully helps only on WILDS PovertyMap (22.4pp → 14.2pp
drop); it's a wash or slightly worse everywhere else — partial, dataset-dependent
support for H4, consistent with only the most tractable of the framework's three
planned components being built so far.

Reproduce any of the three real-world sweeps with:

```bash
PYTHONPATH=src python3 -m ssl_spatial.experiments.real_world_benchmark configs/housing_benchmark.yaml
PYTHONPATH=src python3 -m ssl_spatial.experiments.real_world_benchmark configs/socioeconomic_benchmark.yaml
PYTHONPATH=src python3 -m ssl_spatial.experiments.real_world_benchmark configs/air_quality_benchmark.yaml
```

## Manuscript versions

- `manuscript/main.tex` / `methods.tex` / `results.tex` — original, untouched.
- `manuscript/draft1.tex` (+ `methods_draft1.tex`, `results_draft1.tex`) — adds
  the WILDS PovertyMap benchmark (§2.2.2).
- `manuscript/draft2.tex` (+ `methods_draft2.tex`, `results_draft2.tex`,
  `abstract_draft2.tex`, `discussion_draft2.tex`, `conclusion_draft2.tex`) —
  builds on draft1, adds the three real-world datasets (§2.2.3) and the §2.6
  re-weighting implementation with its H4 test.

## Not yet built (next phases)

- **Spatial GNN / geographically-weighted baseline** (§2.4 "spatially explicit
  models") — needed to test H3; deferred since it needs PyTorch/torch_geometric,
  current scope is sklearn-only.
- **Spatial weighting mechanism and adaptive pseudo-labelling** (§2.6, the
  remaining two components of the distribution-aware framework) — needed to
  re-test H4 with the framework as originally conceived, not just re-weighting.
- **Adaptive/kernel-weighted localised divergence estimator** (§2.3) — the
  fixed-grid MMD estimator's small-sample weakness is now corroborated by a
  population-weighting failure mode on the housing dataset.
- **Formal changepoint analysis** (§2.5 Step 3) — the α≈0.6 breakdown point is
  still a visual read; needs a higher-seed-count re-run (20-30 seeds) and a
  fitted segmented-regression or Bayesian changepoint model.
- **fMoW** — the second WILDS task with a geographic split; PovertyMap was
  chosen instead for tractability (fMoW's archive is much larger).

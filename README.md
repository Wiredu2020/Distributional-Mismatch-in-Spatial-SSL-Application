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
  metrics/divergence.py       KL, Wasserstein, MMD + spatially-localised versions (§6.3)
  metrics/evaluation.py       Accuracy/F1, calibration error, spatial generalisation gap (§6.5 step 2)
  models/baselines.py         Supervised-only, self-training, label propagation (§6.4)
  experiments/
    controlled_experiment.py  Synthetic sweep runner (§6.5 step 1)
    wilds_benchmark.py        WILDS PovertyMap sweep runner (§2.2.2 benchmark eval)
    plot_results.py           Figures for H1/H2 evidence
notebooks/
  benchmark.ipynb              WILDS PovertyMap benchmark walkthrough (§2.2.2 ToDo)
configs/controlled_experiment.yaml   Synthetic sweep definition (edit this to change the experiment)
configs/wilds_benchmark.yaml         WILDS PovertyMap sweep definition
results/controlled_experiment.csv    Output of the last full synthetic run (324 rows)
results/wilds_povertymap_experiment.csv  Output of the last full WILDS benchmark run
data/wilds_povertymap/               Downloaded WILDS PovertyMap metadata + image cache (gitignored)
figures/                              fig1-3 (synthetic), fig4-5 (WILDS) PNGs
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

## Not yet built (next phases)

- **Real-world datasets** (§6.2b: housing, urban socio-economic, environmental
  monitoring) — needs a decision on which public dataset(s) to source.
- **Distribution-aware SSL framework** (§6.6: re-weighting, spatial kernels,
  adaptive pseudo-labelling) — H3/H4 require this to exist before they can be
  tested; the current baselines only test H1/H2.
- **Literature evidence** — the docx's Background section needs citations
  (spatial autocorrelation/non-stationarity in ML, SSL assumption violations,
  domain adaptation under covariate shift). Not something code can produce —
  flag papers as we find them and I'll help work them into the text.
- **Spatial GNN / geographically-weighted baseline** (§6.4 "spatially explicit
  models") — deferred since it needs PyTorch/torch_geometric; current scope is
  sklearn-only per your call above.

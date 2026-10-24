# Reproduction map

Run from the repository root with `PYTHONPATH` set to `src`.

| Configuration | Output specified in YAML |
| --- | --- |
| `air_quality_benchmark.yaml` | `results/air_quality_experiment.csv` |
| `air_quality_benchmark_v2.yaml` | `results/air_quality_experiment_v2.csv` |
| `controlled_experiment.yaml` | `results/controlled_experiment.csv` |
| `controlled_experiment_h4.yaml` | `results/controlled_experiment_h4.csv` |
| `controlled_experiment_sensitivity.yaml` | `results/controlled_experiment_sensitivity.csv` |
| `controlled_experiment_v2.yaml` | `results/controlled_experiment_v2.csv` |
| `h4_ablation_v2.yaml` | `results/h4_ablation_v2.csv` |
| `housing_benchmark.yaml` | `results/housing_experiment.csv` |
| `housing_benchmark_v2.yaml` | `results/housing_experiment_v2.csv` |
| `neural_da_ssl_v2.yaml` | `results/neural_da_ssl_v2.csv` |
| `smoke_test.yaml` | `results/smoke_test.csv` |
| `socioeconomic_benchmark.yaml` | `results/socioeconomic_experiment.csv` |
| `socioeconomic_benchmark_v2.yaml` | `results/socioeconomic_experiment_v2.csv` |
| `spatial_methods_comparison.yaml` | `results/spatial_methods_comparison.csv` |
| `spatial_methods_comparison_v2.yaml` | `results/spatial_methods_comparison_v2.csv` |
| `wilds_benchmark.yaml` | `results/wilds_povertymap_experiment.csv` |
| `wilds_benchmark_h4.yaml` | `results/wilds_povertymap_h4.csv` |
| `wilds_benchmark_v2.yaml` | `results/wilds_povertymap_experiment_v2.csv` |

## Analysis and figures

`localized_mmd_analysis_v2` reads `results/controlled_experiment_v2.csv` and writes the local-MMD comparison and correlation summaries. `changepoint_analysis` reads `results/controlled_experiment_sensitivity.csv` and writes `results/changepoint_analysis.csv`. Both require their input sweeps first.

The plotting module defaults to the older `results/controlled_experiment.csv`. To plot the twenty-seed CSV explicitly:

```bash
python -m ssl_spatial.experiments.plot_results results/controlled_experiment_v2.csv
```

This overwrites the three basic synthetic figures under `figures/`. 

The H4 and neural configurations contain multiple output keys; consult the YAML files and their runners. Neural and geostatistical experiments have not been re-executed in this revision.

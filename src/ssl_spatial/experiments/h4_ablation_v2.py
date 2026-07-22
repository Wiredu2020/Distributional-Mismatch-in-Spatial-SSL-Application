"""H4 ablation study (statistical-rigor + framework-strengthening revision,
following the supervisor review): isolates which component of the
distribution-aware framework (methods.tex Section 2.6) -- spatial weighting,
adaptive pseudo-labelling, or their combination -- is responsible for the
full three-component framework performing worse than re-weighting alone
(supplementary.tex Section S2.1's own open question). Reuses
`spatial_methods_comparison.py`'s generic `run()` / `run_dataset_sweep()`
sweep machinery unchanged; only the config differs
(`configs/h4_ablation_v2.yaml`: 20 seeds, and methods = plain self-training
plus all four distribution-aware variants -- reweight-only, +spatial,
+adaptive, +both).

Usage:
    python -m ssl_spatial.experiments.h4_ablation_v2 configs/h4_ablation_v2.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

from ssl_spatial.experiments.spatial_methods_comparison import run

REPO_ROOT = Path(__file__).resolve().parents[3]

if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else str(REPO_ROOT / "configs" / "h4_ablation_v2.yaml")
    run(config_path)

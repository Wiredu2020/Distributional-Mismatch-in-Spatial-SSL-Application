"""H4 strengthening attempt (supervisor-revision round): evaluates the new
domain-adversarial + consistency PyTorch model
(`ssl_spatial.models.neural_distribution_aware.fit_domain_adversarial_ssl`)
against plain self-training, re-weighting alone, and the existing full
three-component framework, across all 5 datasets at 20 seeds. Reuses
`spatial_methods_comparison.py`'s generic `run()` machinery unchanged; only
`configs/neural_da_ssl_v2.yaml` and the new model differ.

Usage:
    python -m ssl_spatial.experiments.neural_da_ssl_v2 configs/neural_da_ssl_v2.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

from ssl_spatial.experiments.spatial_methods_comparison import run

REPO_ROOT = Path(__file__).resolve().parents[3]

if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else str(REPO_ROOT / "configs" / "neural_da_ssl_v2.yaml")
    run(config_path)

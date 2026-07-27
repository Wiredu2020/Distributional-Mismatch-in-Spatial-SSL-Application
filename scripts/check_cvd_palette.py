"""Verifies ssl_spatial.plotting.PALETTE is colour-vision-deficiency (CVD)
safe for every colour combination that is actually drawn together on one
axis (each METHOD_FAMILIES group, plus the neural-DA comparison), rather
than demanding all 8 palette slots be mutually distinguishable in the
abstract -- no single figure uses more than 5 of them at once.

Applies the Machado, Oliveira & Fialho (2009) linear-RGB simulation
matrices for protanopia, deuteranopia, and tritanopia, then reports the
minimum pairwise Euclidean distance in simulated sRGB space per subset.
A distance below ~35 is flagged as a risk; residual risk is expected to be
covered by METHOD_MARKERS' distinct marker shapes (redundant encoding),
not by colour alone.

Usage:
    python scripts/check_cvd_palette.py
"""
from __future__ import annotations

import itertools

import numpy as np

from ssl_spatial.plotting import METHOD_COLORS, METHOD_FAMILIES

PROTANOPIA = np.array([
    [0.152286, 1.052583, -0.204868],
    [0.114503, 0.786281, 0.099216],
    [-0.003882, -0.048116, 1.051998],
])
DEUTERANOPIA = np.array([
    [0.367322, 0.860646, -0.227968],
    [0.280085, 0.672501, 0.047413],
    [-0.011820, 0.042940, 0.968881],
])
TRITANOPIA = np.array([
    [1.255528, -0.076749, -0.178779],
    [-0.078411, 0.930809, 0.147602],
    [0.004733, 0.691367, 0.303900],
])
SIMULATIONS = {"protanopia": PROTANOPIA, "deuteranopia": DEUTERANOPIA, "tritanopia": TRITANOPIA}
RISK_THRESHOLD = 35.0


def _hex_to_srgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def simulate(hex_colors: list[str], matrix: np.ndarray) -> list[np.ndarray]:
    return [_linear_to_srgb(matrix @ _srgb_to_linear(_hex_to_srgb(h))) * 255 for h in hex_colors]


def check_subset(label: str, hex_colors: list[str]) -> bool:
    print(f"--- {label}: {hex_colors} ---")
    all_ok = True
    for name, matrix in SIMULATIONS.items():
        simulated = simulate(hex_colors, matrix)
        min_dist, worst_pair = min(
            ((np.linalg.norm(a - b), (hex_colors[i], hex_colors[j]))
             for (i, a), (j, b) in itertools.combinations(enumerate(simulated), 2)),
            key=lambda t: t[0],
        )
        ok = min_dist >= RISK_THRESHOLD
        all_ok &= ok
        print(f"  {name:14s} min_dist={min_dist:5.1f}  {worst_pair}  {'OK' if ok else 'RISK'}")
    return all_ok


def main() -> None:
    subsets = {name: [METHOD_COLORS[m] for m in methods] for name, methods in METHOD_FAMILIES.items()}
    subsets["neural_da"] = [METHOD_COLORS[m] for m in
                             ["self_training", "reweighted_self_training", "full_distribution_aware",
                              "domain_adversarial_ssl"]]
    results = {label: check_subset(label, colors) for label, colors in subsets.items()}
    print()
    for label, ok in results.items():
        print(f"{label:16s} {'PASS' if ok else 'FLAGGED (see marker shapes for redundancy)'}")


if __name__ == "__main__":
    main()

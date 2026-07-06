# Pilot findings — controlled synthetic experiment

Source: `results/controlled_experiment.csv` (324 runs: 6 mismatch levels ×
3 autocorrelation lengthscales × 2 non-stationarity levels × 3 seeds × 3
methods). Regenerate with `configs/controlled_experiment.yaml`. Written so
paragraphs can be adapted directly into the manuscript's methodology/results
text — treat as a first pass, not final numbers (N=3 seeds is small; see
Limitations).

## H1 — mismatch degrades performance

Out-of-region accuracy, averaged over lengthscale/non-stationarity/seed, at
minimal (α=0) vs. maximal (α=1) mismatch:

| Method | α=0 | α=1 | Drop |
|---|---|---|---|
| Self-training | 0.874 | 0.786 | 8.8 pp |
| Supervised-only | 0.847 | 0.784 | 6.3 pp |
| Label propagation | 0.859 | 0.811 | 4.9 pp |

All three methods degrade, and degradation is non-linear: accuracy is roughly
flat for α up to ~0.6 and then drops sharply (Figure 1). The spatial
generalisation gap (in-region minus out-of-region accuracy) tells the same
story from a different angle — it hovers near zero for α≤0.6 and jumps to
+0.07–0.10 at α=1 (Figure 2), i.e. models increasingly overfit the region they
were labelled in. **H1 is supported**, with a threshold-like rather than
gradual effect — worth stating explicitly since it argues for identifying a
*breakdown point* rather than assuming monotonic linear decay.

## H2 — cluster/manifold-reliant methods more sensitive: partially supported, nuanced

Self-training (pseudo-labelling, relies on low-density separation) shows the
*largest* absolute drop (8.8 pp) — consistent with H2. But label propagation
(graph/manifold-based) shows the *smallest* drop (4.9 pp), smaller than even
the supervised-only control. That contradicts a naive reading of H2 ("manifold
methods are most fragile"). A plausible explanation: label propagation's
RBF-kernel graph continues to exploit local structure near the labelled region
and effectively degrades toward the supervised solution under severe mismatch,
whereas self-training keeps injecting confidently-wrong pseudo-labels from the
mismatched region, compounding errors. **This nuance — not the clean
hypothesis — is the more interesting result and should be reported as such**,
with the mechanism above offered as a hypothesis for follow-up (e.g., tracking
pseudo-label error rate over training rounds for self-training specifically).

## Non-stationarity alone (no covariate shift) already hurts

At α=0 (labelled and unlabelled drawn from *identical* spatial distributions),
increasing non-stationarity strength from 0 to 1.5 still costs ~4.5 percentage
points across all methods (e.g. supervised-only: 0.869 → 0.824). This
isolates a finding the docx's framing anticipates but doesn't yet distinguish
empirically: **spatial non-stationarity (concept shift) degrades SSL even
when there is no marginal distribution mismatch (covariate shift) at all.**
These are two independent failure modes, not one — the manuscript should keep
them as separate mechanisms in the conceptual framework (Figure X) rather than
folding non-stationarity into "distribution mismatch" as a single construct.

## Section 6.3 metric validation — do KL/Wasserstein/MMD track actual performance loss?

Pooled Pearson correlation with out-of-region accuracy (all 324 runs):

| Metric | r |
|---|---|
| KL divergence (KDE) | −0.67 |
| MMD (RBF) | −0.61 |
| Wasserstein (marginal) | −0.55 |
| Localised MMD (5×5 grid) | −0.19 |

Global divergence metrics are moderately-to-strongly predictive of performance
loss even outside the α sweep (correlation survives, r=−0.55, when restricted
to the α=1.0 subset alone, i.e. it isn't just re-deriving the α parameter).
The **localised** metric is markedly weaker here — likely because at high
global mismatch there are too few grid cells with enough points from both
distributions (small `min_count` after binning), making the localized
estimate noisy rather than informative. This is worth flagging as a concrete
methodological problem to solve in the next iteration (e.g. adaptive binning,
kernel-weighted localisation instead of hard grid cells) rather than as a
negative result about spatial localisation per se.

## Limitations of this pilot (be upfront about these in the write-up)

- Only 3 seeds per cell — good enough to see a real pattern (effects are
  several times larger than seed-to-seed noise) but too few for confidence
  intervals worth reporting.
- Binary classification on 2 synthetic covariates only; real housing/
  environmental data will have more features and different noise structure.
- No spatial GNN or geographically-weighted baseline yet (sklearn-only scope
  for this pass) — §6.4's "spatially explicit models" are still untested.
- H3/H4 (distribution-aware methods outperform / mitigate loss) are untested
  because the distribution-aware framework (§6.6) doesn't exist yet — this
  pilot only establishes that there's a real effect worth building a fix for.

## Suggested next step

Build a minimal version of the §6.6 distribution-aware framework — start with
just the re-weighting component (weight unlabelled points by a density-ratio
estimate of labelled-vs-unlabelled likelihood) layered on top of self-training,
since that's the method showing the largest H1 effect and the most direct
mechanism to fix (down-weight pseudo-labels from regions unlike the labelled
distribution). That gives a first direct test of H4 without requiring the
full spatial-kernel/GNN machinery.

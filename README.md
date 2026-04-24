# Synthetic experiments

Three controlled proof-of-concept runs on synthetic bilevel problems.
Each isolates one specific claim about the algorithm; together they form
the backbone of the paper's Section 4 "Synthetic" block before the
real-data experiments.

## Files

| File | Framework | Upper × Lower | Runtime | Key claim |
|---|---|---|---|---|
| `exp_S1_stiefel_euclidean_pl_game.py` | torch + geoopt | St(10, 3) × ℝ¹² | ~15 s | Classical **Hessian-inverse baseline (HINV) fails** under PL; R-HJFBiO works. |
| `exp_S2_stiefel_spd_pl_game.py` | numpy | St(6, 3) × S⁵₊₊ | ~2 s | Same story on a **non-Euclidean, non-Stiefel** lower manifold (SPD). |
| `exp_S3_stiefel_spd_stochastic.py` | numpy | St(8, 4) × S⁴₊₊ | ~15 s | **SR-HJFBiO** (Algorithm 2) converges; clean variance reduction with B. |

## How to run

```bash
# From this directory
python exp_S1_stiefel_euclidean_pl_game.py     # needs torch + geoopt
python exp_S2_stiefel_spd_pl_game.py           # numpy only
python exp_S3_stiefel_spd_stochastic.py        # numpy only
```

Each script writes a PNG into `outputs/` (created on first run) and
prints a per-run timing and summary to stdout.

## What to look at in each plot

**`exp_S1` — 2×2 grid.** Top row (strongly convex): both methods
converge smoothly, basically indistinguishable. Bottom row (PL-only):
the red `RHGD-HINV` curve hits a "Hessian singular!" annotation and
terminates; the blue `R-HJFBiO` curve continues monotonically. This is
the cleanest demonstration of the paper's core claim.

**`exp_S2` — 3-panel.** Middle panel (gradient norm, log scale) is the
most informative: `RHGD-CG` oscillates between 10⁻¹ and 10⁶ while
`R-HJFBiO` stays at ≈ 0.3. The upper-level objective (left panel) tells
the same story in a less dramatic way: `F ≈ 0.25` for ours, oscillating
in [2, 4] for the baseline.

**`exp_S3` — 3-panel with mean±range bands.** The stochastic curves
(colored, by batch size) overlay the deterministic full-batch reference
(black). Bands come from 5 independent stochastic seeds per batch size.
Expected behaviour: band width shrinks as B grows (the σ² / B term in
Theorem 12).

## Shared dependencies

The two numpy scripts (S2 and S3) import from `../shared/`:
`manifolds.py`, `algorithms.py`, `problems.py`. The torch script (S1)
is self-contained — it uses `geoopt.Stiefel` directly rather than our
in-house implementation.

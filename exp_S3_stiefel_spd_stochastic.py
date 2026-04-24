"""
=============================================================================
SYNTHETIC EXPERIMENT S3 — Stochastic SR-HJFBiO on finite-sum hyper-rep
=============================================================================

Purpose
-------
Verify that the finite-sum variant of our method (SR-HJFBiO, Algorithm 2)
remains stable in the PL-not-SC regime, and observe the textbook
O(sigma^2 / B) variance behaviour as the mini-batch size B grows.

Problem setup
-------------
  Same Stiefel x SPD hyper-representation formulation as Experiment 2 but
  with a finite-sum lower-level loss:

    g(W, M) = (1 / |D_tr|)  sum_{i in D_tr}  (<W^T C_i W, M> - y_i)^2
    f(W, M) = (1 / |D_val|) sum_{i in D_val} (<W^T C_i W, M> - y_i)^2

  Dimensions:  W in St(8, 4),  M in S^4_++,  n_tr = 8,  n_val = 60.
  Since n_tr = 8  <  r(r+1)/2 = 10, the M-Hessian is rank-deficient by
  construction and the lower level is PL-not-SC.

Baselines compared
------------------
  SR-HJFBiO (Ours, B in {1, 2, 4})  : Algorithm 2 of the paper with
                                       three mini-batch sizes, 5 seeds each.
  R-HJFBiO (Ours, full batch)        : Algorithm 1 on the same problem,
                                       serves as the no-noise reference.

  All metrics plotted are the CLEAN full-batch F and ||grad F||,
  evaluated on the iterates every `full_eval_every` iterations -- this
  lets us compare stochastic and deterministic curves on the same grid
  without mini-batch noise contaminating the reported quantities.

Concerns (Andi's review) addressed
----------------------------------
  [Stochastic version]   Demonstrates Algorithm 2 works, matches Thm 12.
  [Non-Euclidean lower]  Lower level is S^4_++.
  [Non-Stiefel second]   SPD manifold, distinct from Stiefel.

Framework
---------
  NumPy + SciPy only.

How to run
----------
    python exp_S3_stiefel_spd_stochastic.py

Output
------
  outputs/synthetic_S3_stochastic.png  -- 3-panel (F, ||grad F||, g) with
                                           per-seed envelopes and a
                                           full-batch reference curve.
=============================================================================
"""

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "shared"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from problems   import HyperRepSPDFiniteSum
from algorithms import r_hjfbio, sr_hjfbio


def run(master_seed=11):
    """5-seed per batch-size run of SR-HJFBiO plus full-batch reference."""
    d, r = 8, 4
    n_tr, n_val = 8, 60       # n_tr < r(r+1)/2 = 10  =>  PL, not SC
    prob = HyperRepSPDFiniteSum(d=d, r=r, n_tr=n_tr, n_val=n_val,
                                 noise=0.05, seed=master_seed)
    print(f"[S3] n_tr={n_tr} < r(r+1)/2={r*(r+1)//2}  ->  PL but not SC")

    # Fixed initial iterate across all runs.
    rng0 = np.random.default_rng(master_seed + 1)
    W0, _ = np.linalg.qr(rng0.standard_normal((d, r)))
    M0 = 1.0 * np.eye(r)
    v0 = np.zeros((r, r))

    T               = 300
    full_eval_every = 5
    batch_sizes     = [1, 2, 4]
    n_seeds         = 5
    hp = dict(gamma=5e-3, lam=1e-2, tau=5e-3,
              mu_clip=0.1, L_clip=50.0,
              delta_eps=1e-3, r_v=3.0)

    results = {B: [] for B in batch_sizes}
    for B in batch_sizes:
        for s in range(n_seeds):
            rng = np.random.default_rng(master_seed + 1000 * B + s)
            t0 = time.time()
            h = sr_hjfbio(prob, W0, M0, v0, T=T,
                          batch_size=B, rng=rng,
                          log_every=1,
                          full_eval_every=full_eval_every, **hp)
            print(f"[S3]   B={B:2d}  seed={s}  ran in "
                  f"{time.time() - t0:.1f}s")
            results[B].append(h)

    t0 = time.time()
    h_full = r_hjfbio(prob, W0, M0, v0, T=T, log_every=1,
                      full_eval_every=full_eval_every, **hp)
    print(f"[S3]   full-batch R-HJFBiO in {time.time() - t0:.1f}s")
    return results, h_full, batch_sizes


def plot(results, h_full, batch_sizes, savepath):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = plt.cm.viridis(np.linspace(0.15, 0.75, len(batch_sizes)))
    x_ref = np.asarray(h_full.eval_iters, dtype=int)

    for c, B in zip(colors, batch_sizes):
        hs = results[B]
        F_mat  = np.stack([np.asarray(h.F_full)      for h in hs], axis=0)
        gx_mat = np.stack([np.asarray(h.grad_x_full) for h in hs], axis=0)
        g_mat  = np.stack([np.asarray(h.g_full)      for h in hs], axis=0)
        x      = np.asarray(hs[0].eval_iters, dtype=int)

        for ax, mat in zip(axes, [F_mat, gx_mat, g_mat]):
            mean = mat.mean(axis=0)
            lo   = mat.min(axis=0)
            hi   = mat.max(axis=0)
            ax.plot(x, mean, color=c, lw=2.0, label=f"SR-HJFBiO  B={B}")
            ax.fill_between(x, lo, hi, color=c, alpha=0.20, linewidth=0)

    axes[0].plot(x_ref, h_full.F_full,      color="black", lw=2.0,
                 label="R-HJFBiO (full)")
    axes[1].plot(x_ref, h_full.grad_x_full, color="black", lw=2.0,
                 label="R-HJFBiO (full)")
    axes[2].plot(x_ref, h_full.g_full,      color="black", lw=2.0,
                 label="R-HJFBiO (full)")

    axes[0].set_xlabel("iteration"); axes[0].set_ylabel("F(W, M)")
    axes[0].set_title("Upper-level objective (full eval)")
    axes[0].grid(True, alpha=.3); axes[0].legend(fontsize=9)

    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel(r"$\|\mathrm{grad}_W F\|$  (clean, full-batch)")
    axes[1].set_yscale("log"); axes[1].set_title(
        "Upper-level Riemannian gradient norm")
    axes[1].grid(True, alpha=.3, which="both"); axes[1].legend(fontsize=9)

    axes[2].set_xlabel("iteration"); axes[2].set_ylabel("g(W, M)")
    axes[2].set_title("Lower-level objective (full eval)")
    axes[2].grid(True, alpha=.3); axes[2].legend(fontsize=9)

    fig.suptitle(r"Synthetic S3: Stochastic SR-HJFBiO on "
                 r"$\mathrm{St}(8,4)\times\mathcal{S}^4_{++}$, "
                 r"B in {1, 2, 4}, mean $\pm$ range over 5 seeds",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(savepath, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs("outputs", exist_ok=True)
    results, h_full, batch_sizes = run()
    out = "outputs/synthetic_S3_stochastic.png"
    plot(results, h_full, batch_sizes, out)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()

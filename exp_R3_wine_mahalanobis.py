"""
=============================================================================
REAL-WORLD EXPERIMENT R3 — Mahalanobis metric learning on UCI Wine
=============================================================================

Purpose
-------
Evaluate R-HJFBiO on a canonical classical-ML task: supervised Mahalanobis
metric learning from labelled pair information.  This formulation sits at
the heart of a large body of metric-learning work (LMNN, ITML, NCA, ...)
and is a natural home for Riemannian optimisation on SPD: the metric
matrix M lives on S^d_++.

Problem setup
-------------
Data: `sklearn.datasets.load_wine` (178 samples, 13 standardised chemical
features, 3 classes).  We restrict to two classes and form labelled sample
PAIRS (i, j) with a rank-1 pair-difference "sensing matrix" and a target
distance class-label:

    C_ij = (x_i - x_j)(x_i - x_j)^T  in  S^{13}   (rank 1)
    y_ij = +1  if x_i and x_j come from DIFFERENT classes,
    y_ij = -1  if they come from the SAME class.

Bilevel formulation (Stiefel x SPD; same bilinear form as experiments R1, R2):

    min_{W in St(13, 5)} F(W, M) = (1/n_val) sum_ij (<W^T C_ij W, M> - y_ij)^2
    s.t. M in argmin_{S^5_++} (1/n_tr) sum_ij (<W^T C_ij W, M> - y_ij)^2

W projects the raw feature space to a 5-dim subspace; M is the Mahalanobis
metric in that subspace.  The scalar score <W^T C_ij W, M> is a learnt
squared distance; pushing same-class pairs toward -1 and different-class
pairs toward +1 realises the classical contrastive metric-learning objective.

Because each C_ij has rank 1, the Hessian of g in M has rank at most
n_tr < r(r+1)/2 = 15, independent of how rich the C_ij subspace is.
We use n_tr_pairs = 14  <  15  ==>  PL-but-not-SC is automatic.
n_val_pairs = 800 held-out pairs.

Baselines compared
------------------
  RHGD-CG (Han et al.)  : inner-loop RGD + CG solve of Hess_M g v = grad f.
  R-HJFBiO (Ours)       : Algorithm 1.

Expected behaviour
------------------
Both methods hit 100% training accuracy -- memorisation of 14 pairs is
easy.  The interesting phenomena:
  * R-HJFBiO descends smoothly and monotonically.
  * RHGD-CG exhibits FIVE late-stage CG-breakdown events (visible
    gradient-norm spikes), but happens to generalise better in the
    particular seed.  We report this honestly -- it's a useful reminder
    that the paper's contribution is CONVERGENCE, not a generalisation
    recipe, and that implicit regularisation from instability can
    sometimes help on severely underdetermined problems.

Concerns (Andi's review) addressed
----------------------------------
  [Real data]            UCI Wine (178 samples, 13 features).
  [Non-Euclidean lower]  S^5_++ affine-invariant metric.
  [Non-Stiefel second]   SPD manifold, natively the setting of metric
                         learning.
  [Classical ML fit]     Mahalanobis metric learning is one of the
                         longest-standing Riemannian-optimisation use
                         cases.

Framework
---------
  NumPy + SciPy + sklearn.

How to run
----------
    python exp_R3_wine_mahalanobis.py

Output
------
  outputs/real_R3_wine.png  -- 3-panel (F, ||grad F||, g), val acc in legend.
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
from algorithms import r_hjfbio, rhgd_cg
from real_data  import wine_pair_differences


def classification_accuracy(prob, W, M, split="val"):
    if split == "val":
        C, y = prob.C_val, prob.y_val
    else:
        C, y = prob.C_tr, prob.y_tr
    Proj = np.einsum("ipq,pa,qb->iab", C, W, W)
    s    = np.einsum("iab,ab->i", Proj, M)
    s_mid = (0.5 * (s[y > 0].mean() + s[y < 0].mean())
             if len(set(y)) > 1 else 0.0)
    pred = np.where(s > s_mid, +1.0, -1.0)
    return float((pred == y).mean())


def plot_pair(hist_ours, hist_cg, title, savepath,
              acc_ours=None, acc_cg=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    lbl_ours = "R-HJFBiO (ours)"
    lbl_cg   = "RHGD-CG (Han et al.)"
    if acc_ours is not None:
        lbl_ours = (f"R-HJFBiO (ours)   best-iter val acc "
                    f"{acc_ours[0]*100:.1f}%  (tr {acc_ours[1]*100:.0f}%)")
    if acc_cg is not None:
        lbl_cg = (f"RHGD-CG (Han et al.)   best-iter val acc "
                  f"{acc_cg[0]*100:.1f}%  (tr {acc_cg[1]*100:.0f}%)")

    axes[0].plot(hist_ours.F, label=lbl_ours, color="C0", lw=2)
    axes[0].plot(hist_cg.F,   label=lbl_cg,   color="C3", lw=2, ls="--")
    axes[0].set_xlabel("iteration"); axes[0].set_ylabel("F (validation)")
    axes[0].set_title("Upper-level objective"); axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=.3)

    axes[1].plot(hist_ours.grad_x, label="R-HJFBiO (ours)",
                 color="C0", lw=2)
    axes[1].plot(hist_cg.grad_x,   label="RHGD-CG",
                 color="C3", lw=2, ls="--")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel(r"$\|\mathrm{grad}_W F\|$")
    axes[1].set_title("Upper-level Riemannian gradient norm")
    axes[1].set_yscale("log"); axes[1].legend()
    axes[1].grid(True, alpha=.3, which="both")

    axes[2].plot(hist_ours.g, label="R-HJFBiO (ours)", color="C0", lw=2)
    axes[2].plot(hist_cg.g,   label="RHGD-CG",         color="C3", lw=2, ls="--")
    axes[2].set_xlabel("iteration"); axes[2].set_ylabel("g (training)")
    axes[2].set_title("Lower-level objective"); axes[2].legend()
    axes[2].grid(True, alpha=.3)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(savepath, dpi=130, bbox_inches="tight")
    plt.close(fig)


def run(seed=2):
    data = wine_pair_differences(classes=(0, 1), n_tr_pairs=14,
                                  n_val_pairs=800, seed=seed)
    d, r = data["d"], 5
    prob = HyperRepSPDFiniteSum.from_data(
        data["C_tr"], data["y_tr"], data["C_val"], data["y_val"],
        d=d, r=r)
    print(f"[R3] Wine: d={d}, r={r}, n_tr_pairs={len(data['y_tr'])}, "
          f"n_val_pairs={len(data['y_val'])}")

    rng = np.random.default_rng(seed + 1)
    W0, _ = np.linalg.qr(rng.standard_normal((d, r)))
    M0 = np.eye(r); v0 = np.zeros((r, r))

    T = 600
    t0 = time.time()
    hist_ours, final_ours, best_ours = r_hjfbio(
        prob, W0, M0, v0, T=T,
        gamma=1e-2, lam=2e-2, tau=1e-2,
        mu_clip=0.1, L_clip=50.0,
        delta_eps=1e-3, r_v=3.0, log_every=1,
        return_final=True, track_best=True)
    print(f"[R3] R-HJFBiO ran in {time.time() - t0:.1f}s")

    t0 = time.time()
    hist_cg, final_cg, best_cg = rhgd_cg(
        prob, W0, M0, K=T,
        eta_x=1e-2, eta_y=2e-2,
        inner_steps=5, cg_iters=20,
        cg_tol=1e-10, log_every=1,
        return_final=True, track_best=True)
    print(f"[R3] RHGD-CG  ran in {time.time() - t0:.1f}s")

    def pair_acc(W, M):
        return (classification_accuracy(prob, W, M, "val"),
                classification_accuracy(prob, W, M, "tr"))
    acc_ours_best  = pair_acc(best_ours[0],  best_ours[1])
    acc_cg_best    = pair_acc(best_cg[0],    best_cg[1])
    acc_ours_final = pair_acc(final_ours[0], final_ours[1])
    acc_cg_final   = pair_acc(final_cg[0],   final_cg[1])

    print(f"[R3] R-HJFBiO final acc val={acc_ours_final[0]*100:.1f}%  "
          f"tr={acc_ours_final[1]*100:.0f}%    "
          f"best (iter {best_ours[4]:>3d}) val={acc_ours_best[0]*100:.1f}%  "
          f"tr={acc_ours_best[1]*100:.0f}%")
    print(f"[R3] RHGD-CG  final acc val={acc_cg_final[0]*100:.1f}%  "
          f"tr={acc_cg_final[1]*100:.0f}%    "
          f"best (iter {best_cg[4]:>3d}) val={acc_cg_best[0]*100:.1f}%  "
          f"tr={acc_cg_best[1]*100:.0f}%")

    return hist_ours, hist_cg, acc_ours_best, acc_cg_best


def main():
    os.makedirs("outputs", exist_ok=True)
    hist_ours, hist_cg, acc_ours_best, acc_cg_best = run(seed=2)
    out = "outputs/real_R3_wine.png"
    plot_pair(hist_ours, hist_cg,
              r"Real-world R3: UCI Wine Mahalanobis metric learning,  "
              r"$\mathrm{St}(13, 5)\times\mathcal{S}^5_{++}$",
              out, acc_ours=acc_ours_best, acc_cg=acc_cg_best)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()

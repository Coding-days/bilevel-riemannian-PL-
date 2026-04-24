"""
=============================================================================
REAL-WORLD EXPERIMENT R1 — SPD hyper-representation on sklearn digits
=============================================================================

Purpose
-------
Run the Han-et-al. NeurIPS-2024 shallow hyper-representation formulation
(Framework paper, Section 4.2) on REAL image data, WITHOUT the lam||beta||^2
regularizer their theory requires.  Without regularization and with a small
training set, the lower-level Hessian is rank-deficient and the problem is
PL but not geodesically strongly convex.  This is the regime R-HJFBiO is
designed for.

Problem setup
-------------
Data: `sklearn.datasets.load_digits` (1797 real 8x8 handwritten digit images,
10 classes).  We restrict to two classes (0 vs 1) to get binary targets
y_i in {+1, -1}.  For each image, we compute a 5x5 SPD region covariance
descriptor (Tuzel, Porikli, Meer; ECCV 2006):

    f(x, y) = [x, y, I(x, y), |I_x(x, y)|, |I_y(x, y)|]  in R^5
    A_i = (1/N_px) sum_{(x,y)} (f(x,y) - mean)(f(x,y) - mean)^T + eps * I

Bilevel formulation (same bilinear form as Han et al., without regularizer):
    min_{W in St(5, 4)} F(W, M) = (1/n_val) sum_i (<W^T A_i W, M> - y_i)^2
    s.t. M in argmin_{S^4_++} (1/n_tr) sum_i (<W^T A_i W, M> - y_i)^2

n_tr = 6  <  r(r+1)/2 = 10  ==>  PL-but-not-SC lower level
n_val = 200

Baselines compared
------------------
  RHGD-CG (Han et al.)  : inner-loop RGD + CG solve of  Hess_M g  v = grad f.
  R-HJFBiO (Ours)       : Algorithm 1.

Both methods also report the best-iterate (argmin F on the validation loss)
classification accuracy obtained by reading off the sign of the bilinear
score <W^T A_i W, M>.

Concerns (Andi's review) addressed
----------------------------------
  [Real data]            sklearn handwritten digits, 360 samples.
  [Non-Euclidean lower]  S^4_++ affine-invariant.
  [Non-Stiefel second]   SPD manifold.
  [Application fit]      Region covariance descriptors are the canonical
                         SPD representation for images (SPDNet, 2017+).

Framework
---------
  NumPy + SciPy + sklearn.  (Requires the shared/ modules.)

How to run
----------
    python exp_R1_digits_hyperrep.py

Output
------
  outputs/real_R1_digits.png  -- 3-panel (F, ||grad F||, g), with final
                                validation accuracy in the legend.
  Console                    -- Train/val accuracy for final and
                                best-F iterates.
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
from real_data  import digits_region_covariances


# ----------------------------------------------------------------------------
# Classification accuracy on val or tr split
# ----------------------------------------------------------------------------
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


def run(seed=0):
    data = digits_region_covariances(classes=(0, 1), n_tr=6, n_val=200,
                                      seed=seed)
    d, r = data["d"], 4
    prob = HyperRepSPDFiniteSum.from_data(
        data["C_tr"], data["y_tr"], data["C_val"], data["y_val"],
        d=d, r=r)
    print(f"[R1] digits: d={d}, r={r}, n_tr={len(data['y_tr'])}, "
          f"n_val={len(data['y_val'])}")

    rng = np.random.default_rng(seed + 1)
    W0, _ = np.linalg.qr(rng.standard_normal((d, r)))
    M0 = np.eye(r); v0 = np.zeros((r, r))

    T = 600
    t0 = time.time()
    hist_ours, final_ours, best_ours = r_hjfbio(
        prob, W0, M0, v0, T=T,
        gamma=2e-2, lam=3e-2, tau=2e-2,
        mu_clip=0.1, L_clip=50.0,
        delta_eps=1e-3, r_v=3.0, log_every=1,
        return_final=True, track_best=True)
    print(f"[R1] R-HJFBiO ran in {time.time() - t0:.1f}s")

    t0 = time.time()
    hist_cg, final_cg, best_cg = rhgd_cg(
        prob, W0, M0, K=T,
        eta_x=2e-2, eta_y=3e-2,
        inner_steps=5, cg_iters=20,
        cg_tol=1e-10, log_every=1,
        return_final=True, track_best=True)
    print(f"[R1] RHGD-CG  ran in {time.time() - t0:.1f}s")

    def pair_acc(W, M):
        return (classification_accuracy(prob, W, M, "val"),
                classification_accuracy(prob, W, M, "tr"))
    acc_ours_final = pair_acc(final_ours[0], final_ours[1])
    acc_ours_best  = pair_acc(best_ours[0],  best_ours[1])
    acc_cg_final   = pair_acc(final_cg[0],   final_cg[1])
    acc_cg_best    = pair_acc(best_cg[0],    best_cg[1])

    print(f"[R1] R-HJFBiO final acc val={acc_ours_final[0]*100:.1f}%  "
          f"tr={acc_ours_final[1]*100:.0f}%    "
          f"best (iter {best_ours[4]:>3d}) val={acc_ours_best[0]*100:.1f}%  "
          f"tr={acc_ours_best[1]*100:.0f}%")
    print(f"[R1] RHGD-CG  final acc val={acc_cg_final[0]*100:.1f}%  "
          f"tr={acc_cg_final[1]*100:.0f}%    "
          f"best (iter {best_cg[4]:>3d}) val={acc_cg_best[0]*100:.1f}%  "
          f"tr={acc_cg_best[1]*100:.0f}%")

    return (prob, hist_ours, hist_cg,
            acc_ours_final, acc_cg_final,
            acc_ours_best,  acc_cg_best)


def main():
    os.makedirs("outputs", exist_ok=True)
    (_, hist_ours, hist_cg,
     _af, _cf, acc_ours_best, acc_cg_best) = run(seed=0)
    out = "outputs/real_R1_digits.png"
    plot_pair(hist_ours, hist_cg,
              r"Real-world R1: handwritten digits region covariances,  "
              r"$\mathrm{St}(5, 4)\times\mathcal{S}^4_{++}$",
              out,
              acc_ours=acc_ours_best, acc_cg=acc_cg_best)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()

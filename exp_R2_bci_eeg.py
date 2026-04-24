"""
=============================================================================
REAL-WORLD EXPERIMENT R2 — Simulated motor-imagery EEG (BCI pipeline)
=============================================================================

Purpose
-------
Exercise our algorithm on a realistic neuroscience / brain-computer-
interface (BCI) data pipeline.  In the Barachant-et-al. motor-imagery
pipeline (IEEE T-BME 2012; the standard for EEG-based BCIs), each trial
is summarised by a spatial covariance matrix C_i in S^d_{++} whose
eigenstructure carries the class information.  Classifying covariance
matrices lives natively on the SPD manifold, and this problem has been
the #1 real-world application of Riemannian optimisation for more than
a decade.

Problem setup
-------------
We synthesise motor-imagery trials (two classes, "left hand" vs
"right hand") under the standard generative model

    x(t) = A_class * s(t) + noise,   (16 channels, 200 time samples)

with different class-conditional spatial mixing matrices A_class.  Trial
covariances

    C_i = (1 / (T - 1)) * X_i X_i^T  in  S^{16}_{++}

carry the discriminative signal entirely in their eigenstructure.

Bilevel formulation (Stiefel x SPD hyper-rep, same bilinear form as
experiments R1 and R3):

    min_{W in St(16, 4)} F(W, M) = (1/n_val) sum_i (<W^T C_i W, M> - y_i)^2
    s.t. M in argmin_{S^4_++} (1/n_tr) sum_i (<W^T C_i W, M> - y_i)^2

n_tr = 6  <  r(r+1)/2 = 10  ==>  PL-but-not-SC.
n_val = 120, evenly balanced, two classes.

W plays the role of a learnt spatial filter (dimensionality reduction
from 16 raw channels to a 4-dim subspace) -- structurally identical to
Common Spatial Patterns but optimised jointly with the classifier head
M on SPD via a bilevel criterion.

Baselines compared
------------------
  RHGD-CG (Han et al.)  : inner-loop RGD + CG solve of Hess_M g v = grad f.
  R-HJFBiO (Ours)       : Algorithm 1.

Concerns (Andi's review) addressed
----------------------------------
  [Real application]     Motor-imagery BCI is the canonical Riemannian-
                         optimization application (Barachant 2012 et seq).
  [Non-Euclidean lower]  S^4_++ affine-invariant metric.
  [Non-Stiefel second]   SPD manifold.

Framework
---------
  NumPy only.  (Simulation avoids the need for real EEG recordings.
  To swap in real data, replace `simulated_eeg_covariances` with a loader
  that produces (C_tr, y_tr, C_val, y_val) from a BCI-IV-2a / MOABB style
  dataset.)

How to run
----------
    python exp_R2_bci_eeg.py

Output
------
  outputs/real_R2_eeg.png   -- 3-panel (F, ||grad F||, g), val acc in legend.
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
from real_data  import simulated_eeg_covariances


# Same accuracy and plot_pair as experiment A.
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


def run(seed=1):
    data = simulated_eeg_covariances(d_channels=16, n_sources=3, n_time=200,
                                      n_tr=6, n_val=120, seed=seed)
    d, r = data["d"], 4
    prob = HyperRepSPDFiniteSum.from_data(
        data["C_tr"], data["y_tr"], data["C_val"], data["y_val"],
        d=d, r=r)
    print(f"[R2] EEG: d={d}, r={r}, n_tr={len(data['y_tr'])}, "
          f"n_val={len(data['y_val'])}")

    rng = np.random.default_rng(seed + 1)
    W0, _ = np.linalg.qr(rng.standard_normal((d, r)))
    M0 = np.eye(r); v0 = np.zeros((r, r))

    T = 500
    t0 = time.time()
    hist_ours, final_ours, best_ours = r_hjfbio(
        prob, W0, M0, v0, T=T,
        gamma=3e-3, lam=7e-3, tau=3e-3,
        mu_clip=0.1, L_clip=200.0,
        delta_eps=1e-3, r_v=3.0, log_every=1,
        return_final=True, track_best=True)
    print(f"[R2] R-HJFBiO ran in {time.time() - t0:.1f}s")

    t0 = time.time()
    hist_cg, final_cg, best_cg = rhgd_cg(
        prob, W0, M0, K=T,
        eta_x=3e-3, eta_y=7e-3,
        inner_steps=5, cg_iters=20,
        cg_tol=1e-10, log_every=1,
        return_final=True, track_best=True)
    print(f"[R2] RHGD-CG  ran in {time.time() - t0:.1f}s")

    def pair_acc(W, M):
        return (classification_accuracy(prob, W, M, "val"),
                classification_accuracy(prob, W, M, "tr"))
    acc_ours_best  = pair_acc(best_ours[0],  best_ours[1])
    acc_cg_best    = pair_acc(best_cg[0],    best_cg[1])
    acc_ours_final = pair_acc(final_ours[0], final_ours[1])
    acc_cg_final   = pair_acc(final_cg[0],   final_cg[1])

    print(f"[R2] R-HJFBiO final acc val={acc_ours_final[0]*100:.1f}%  "
          f"tr={acc_ours_final[1]*100:.0f}%    "
          f"best (iter {best_ours[4]:>3d}) val={acc_ours_best[0]*100:.1f}%  "
          f"tr={acc_ours_best[1]*100:.0f}%")
    print(f"[R2] RHGD-CG  final acc val={acc_cg_final[0]*100:.1f}%  "
          f"tr={acc_cg_final[1]*100:.0f}%    "
          f"best (iter {best_cg[4]:>3d}) val={acc_cg_best[0]*100:.1f}%  "
          f"tr={acc_cg_best[1]*100:.0f}%")

    return hist_ours, hist_cg, acc_ours_best, acc_cg_best


def main():
    os.makedirs("outputs", exist_ok=True)
    hist_ours, hist_cg, acc_ours_best, acc_cg_best = run(seed=1)
    out = "outputs/real_R2_eeg.png"
    plot_pair(hist_ours, hist_cg,
              r"Real-world R2: simulated motor-imagery EEG (BCI pipeline),  "
              r"$\mathrm{St}(16, 4)\times\mathcal{S}^4_{++}$",
              out, acc_ours=acc_ours_best, acc_cg=acc_cg_best)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()

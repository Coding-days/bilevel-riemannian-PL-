"""
=============================================================================
SYNTHETIC EXPERIMENT S2 — Stiefel x SPD bilevel PL game
=============================================================================

Purpose
-------
The Stiefel-x-SPD extension of Experiment 1.  Both manifolds are now
non-trivial.  This directly addresses the reviewer comment that the lower
level should sit on a manifold beyond Euclidean space, and in particular
should be non-Stiefel.  The lower level now lives on S^p_{++} with the
affine-invariant metric.

Problem setup
-------------
    min_{W in St(d_x, r)}  f(W, M*(W)) = 0.5 * ||W - W_tar||^2_F
                                        + alpha * <R_coup, M*(W) - M_tar>
    s.t.                   M*(W) = argmin_{M in S^p_++}  g(W, M)

    g(W, M) = 0.5 * sum_{i=1}^{n_sense} (<C_i, M> - b_i(W))^2
    b_i(W) = <c_i, W> + <C_i, M_tar>
    C_i rank-1 symmetric sensing matrices; n_sense = 4 << p(p+1)/2 = 15.

The rank deficiency of the sensing operator enforces
            rank(Hess_M g) <= 4  <<  15 = dim(S^p),
so the lower-level problem is PL but NOT geodesically strongly convex
(empirical PL constant approx 0.97, confirmed by diagnose_pl.py).

Baselines compared
------------------
  RHGD-CG (Han et al.)  : Riemannian hypergradient descent with CG solve
                          of  Hess_M g  v = grad_M f.  Fails catastrophically
                          when Hess_M g is rank-deficient.
  R-HJFBiO (Ours)       : spectral clipping + finite-difference surrogates.
                          Stable on this problem.

Concerns (Andi's review) addressed
----------------------------------
  [Non-Euclidean lower]  Lower level is S^p_++ (affine-invariant).
  [Non-Stiefel second manifold]  SPD is distinct from Stiefel.
  [Clear dominance]      CG oscillates ||grad F|| between 10^-1 and 10^6;
                         R-HJFBiO stays at ~0.3 throughout.

Framework
---------
  NumPy + SciPy only.  (Requires: numpy, scipy, matplotlib, and the
  shared/ modules: manifolds.py, algorithms.py, problems.py.)

How to run
----------
    python exp_S2_stiefel_spd_pl_game.py

Output
------
  outputs/synthetic_S2_stiefel_spd.png  -- 3-panel figure  (F, ||grad F||, g)
  Console                                -- per-run timings.
=============================================================================
"""

import os
import sys
import time

# Make shared/ importable regardless of where the script is invoked from.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "shared"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from problems   import BilevelPLGameStiefelSPD
from algorithms import r_hjfbio, rhgd_cg
from manifolds  import sym


def run():
    """Deterministic comparison on a rank-deficient Stiefel x SPD PL game."""
    d_x, r, p = 6, 3, 5
    n_sense   = 4                                     # << p(p+1)/2 = 15
    seed      = 42

    prob = BilevelPLGameStiefelSPD(d_x, r, p, n_sense=n_sense,
                                    alpha=0.1, seed=seed)

    # Shared initialization for fairness.
    rng = np.random.default_rng(seed)
    W0, _ = np.linalg.qr(rng.standard_normal((d_x, r)))
    M0    = np.eye(p) + 0.1 * sym(rng.standard_normal((p, p)))
    v0    = np.zeros((p, p))

    T = 400

    t0 = time.time()
    hist_ours = r_hjfbio(prob, W0, M0, v0, T=T,
                         gamma=5e-3, lam=5e-3, tau=5e-3,
                         mu_clip=0.05, L_clip=float(prob.Lg_euc) + 1.0,
                         delta_eps=1e-3, r_v=5.0,
                         log_every=1, verbose=False)
    print(f"[S2] R-HJFBiO ran in {time.time() - t0:.1f}s")

    t0 = time.time()
    hist_cg = rhgd_cg(prob, W0, M0, K=T,
                       eta_x=5e-3, eta_y=5e-3,
                       inner_steps=5, cg_iters=30,
                       cg_tol=1e-10, log_every=1, verbose=False)
    print(f"[S2] RHGD-CG  ran in {time.time() - t0:.1f}s")
    return hist_ours, hist_cg


def plot(hist_ours, hist_cg, savepath):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(hist_ours.F, label="R-HJFBiO (ours)", color="C0", lw=2)
    axes[0].plot(hist_cg.F,   label="RHGD-CG (Han et al.)",
                 color="C3", lw=2, ls="--")
    axes[0].set_xlabel("iteration"); axes[0].set_ylabel("F(W, M)")
    axes[0].set_title("Upper-level objective"); axes[0].legend()
    axes[0].grid(True, alpha=.3)

    axes[1].plot(hist_ours.grad_x, label="R-HJFBiO (ours)",
                 color="C0", lw=2)
    axes[1].plot(hist_cg.grad_x,   label="RHGD-CG (Han et al.)",
                 color="C3", lw=2, ls="--")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel(r"$\|\mathrm{grad}_W F\|$")
    axes[1].set_title("Upper-level Riemannian gradient norm")
    axes[1].set_yscale("log"); axes[1].legend()
    axes[1].grid(True, alpha=.3, which="both")

    axes[2].plot(hist_ours.g, label="R-HJFBiO (ours)", color="C0", lw=2)
    axes[2].plot(hist_cg.g,   label="RHGD-CG (Han et al.)",
                 color="C3", lw=2, ls="--")
    axes[2].set_xlabel("iteration"); axes[2].set_ylabel("g(W, M)")
    axes[2].set_title("Lower-level objective"); axes[2].legend()
    axes[2].grid(True, alpha=.3)

    fig.suptitle(r"Synthetic S2: bilevel PL game on "
                 r"$\mathrm{St}(6,3)\times\mathcal{S}^5_{++}$, "
                 r"rank-deficient PL lower level",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(savepath, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs("outputs", exist_ok=True)
    hist_ours, hist_cg = run()
    out = "outputs/synthetic_S2_stiefel_spd.png"
    plot(hist_ours, hist_cg, out)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()

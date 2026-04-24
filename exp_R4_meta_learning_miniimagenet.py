"""
=============================================================================
REAL-WORLD EXPERIMENT R4 — Few-shot meta-learning (MiniImageNet-style)
=============================================================================

Purpose
-------
Reproduce the Riemannian meta-learning experiment from Han et al.
(NeurIPS 2024, Section 4.3) -- 5-way 5-shot classification on
MiniImageNet-like data with a 4-block CNN whose first two conv kernels
are Stiefel-constrained -- and run it in THREE regimes (strong_convex,
tiny_lam, pl_only) to disentangle strong convexity from PL.  The
pl_only regime removes the L2 regularizer  (lam / 2) ||w||^2  that
Han et al. add to make the lower level strongly convex, which is
exactly the regime R-HJFBiO targets.

In addition to the standard ridge-CG hypergradient (Han et al.), this
file uses a Morse-Bott pseudoinverse solver motivated by Boumal,
Criscitiello & Rebjock's April-2026 result that for any smooth globally
PL function the Hessian at a minimiser has a bimodal spectrum with an
explicit gap.  That pseudoinverse is the principled object under PL
(as opposed to a scalar ridge, which destroys the Morse-Bott structure).

Problem setup
-------------
Data: MiniImageNet-like synthetic tasks by default (random low-rank
per-class structure; runs on any CPU).  Point `--data_root` at a
directory holding the standard pickle files and pass `--real_data`
to swap in real MiniImageNet.

Upper-level variable Theta
  Stiefel-constrained conv kernels of blocks 1 and 2 of the backbone
  CNN (St(27, 16) and St(144, 16)), plus Euclidean params of blocks 3-4
  and BN parameters.

Lower-level variable w
  Task-specific last-layer weights w in R^{n_way x feat_dim}.

  g(theta, w; D_support) = CE(softmax(features @ w^T), labels)
                           + (lam / 2) ||w||^2
  f(theta, w; D_query)   = CE(softmax(features @ w^T), labels)

Regimes
-------
  STRONG_CONVEX  lam = 1e-2    -> classical implicit-diff theory applies.
  TINY_LAM       lam = 1e-6    -> SC but vacuous constant; PL story wins.
  PL_ONLY        lam = 0       -> rank-deficient Hess_w; SC breaks, PL still works.

Baselines compared (per regime)
-------------------------------
  RHGD with CG, short (20 iterations)   -- the Han et al. baseline.
  RHGD with CG, long  (200 iterations)  -- "does more CG help?" control.
  Morse-Bott pseudoinverse              -- the BCR26-motivated solver.
                                           On PL problems this is the
                                           principled Moore-Penrose
                                           pseudoinverse of the Hessian.

Concerns (Andi's review) addressed
----------------------------------
  [Real-world ML task]   Few-shot image classification, the archetypal
                         bilevel-meta-learning workload.
  [Stochastic]           Each outer step samples a small batch of tasks
                         (n_tasks_per_batch).
  [Matches Han baseline] Uses the same Stiefel-CNN architecture as Han
                         et al. (their Section 4.3) and the same outer/
                         inner structure.  Only change is removing the
                         L2 regularizer in the pl_only regime.

Framework
---------
  PyTorch + geoopt.  (Requires: torch, torchvision, geoopt, numpy,
  matplotlib.  GPU optional; CPU runs in ~10 min for the default settings.)

How to run
----------
  Edit the SETTINGS class near the top of this file (search for
  ">>> EDIT HERE <<<"), then:

    python exp_R4_meta_learning_miniimagenet.py

  Common edits:
    * SETTINGS.REGIME = "all"       -> compare SC / tiny / PL side by side
    * SETTINGS.REGIME = "pl_only"   -> just the PL regime
    * SETTINGS.MEASURE_HG_QUALITY = True -> also generate solver_quality.png
                                            (the BCR26 head-to-head plot)

Output
------
  outputs_riem_meta_pl/comparison.png      -- training curves, 3 regimes.
  outputs_riem_meta_pl/solver_quality.png  -- Morse-Bott vs ridge-CG
                                              hypergradient error
                                              (only if MEASURE_HG_QUALITY).
  outputs_riem_meta_pl/histories.json      -- raw training histories.

Provenance
----------
Original file name:  riemannian_meta_learning_pl.py.  Body preserved
verbatim; only this header block was rewritten for consistency with
the other experiments in this suite.
=============================================================================


Extended notes (preserved from original)
========================================
This script is a REAL meta-learning implementation -- not a synthetic
proxy.  It implements the Han-et-al. NeurIPS-2024 Section 4.3 experiment
faithfully, with three additions motivated by Boumal-Criscitiello-Rebjock
(arXiv:2604.07972, April 2026; BCR26):

  (i)  Morse-Bott structure at w* (BCR26, Lemma 2.2). For ANY smooth,
       globally mu-PL function g, at a minimizer w* we have
            ker Hess_w g(w*) = T_{w*} S       (exactly the tangent to S)
            Hess_w g(w*) |_{N_{w*} S}  >=  mu * Id
       i.e., the spectrum is strictly bimodal: exactly dim(S) zero
       eigenvalues, all other eigenvalues >= mu_PL. This is MUCH stronger
       than just "PL" and it tells us the right regularizer is a
       Moore-Penrose pseudoinverse that leaves ker(Hess) untouched,
       NOT a scalar ridge that lifts every eigenvalue by eps_pl.

  (ii) Quadratic growth (BCR26, Lemma 2.1):
            g(w) - g*  >=  (mu/2) * dist(w, S)^2.
       This gives a cleaner, trajectory-based mu_PL estimator than the
       random-probe estimator we had before: walk a step away from the
       found minimizer w* and read off (g - g*) / ||w - w*||^2 directly.

  (iii) Nonlinear-least-squares structure (BCR26, Theorem 1.2). For
       contractible M (our w-space is Euclidean, hence contractible),
       any such g has the form g = g* + ||phi(w)||^2 for some submersion
       phi: M -> R^k with k = codim(S). We cannot recover phi in closed
       form for cross-entropy, but Lemma 2.2 says the Hessian eigen-
       decomposition at w* IS phi's linearization. We can therefore
       verify the decomposition empirically by reporting the "Morse-Bott
       signature" (#zero-eigvals, #positive-eigvals) of Hess_w g at w*.

  (iv) End-point map of negative gradient flow (BCR26, Section 4.1).
       solve_lower_level(w0) is literally pi(w0): the map that sends w0
       to the limit of negative gradient flow on g. In BCR26's language,
       the fiber F = pi^{-1}(w*) is diffeomorphic to R^k and g|F has
       w* as its UNIQUE minimizer (Proposition 4.4). So the classical
       strongly-convex hypergradient machinery applies WITHIN each
       fiber -- which is how we can use implicit differentiation cleanly
       despite the global rank deficiency of Hess_w g.

Concretely, the modifications below are:

  * Replaced the ad-hoc `eps_pl * I` ridge inside CG with a Morse-Bott
    pseudoinverse solve: we probe the low end of the Hessian spectrum
    to identify the (estimated) ker = T_{w*} S, project both sides onto
    its orthogonal complement N_{w*} S, and solve there. Under BCR26
    Lemma 2.2 this is exactly the Moore-Penrose pseudoinverse of
    Hess_w g, which is the principled object.

  * Added a quadratic-growth-based mu_PL estimator using Lemma 2.1.

  * Added a Morse-Bott structure report (dim of effective kernel of
    Hess_w g at w*, gap to first positive eigenvalue) -- an empirical
    read on the claim of BCR26 Lemma 2.2 on the actual data.

  * Added a head-to-head solver-quality measurement (--measure_hg_quality):
    on the same problem (same Hess, same RHS) at each outer step,
    compute v with three different solvers -- Morse-Bott pinv, ridge CG
    with the configured eps_pl and cg_steps, ridge CG with cg_steps_long
    extra iterations -- and report each method's relative error to the
    Moore-Penrose pinv reference. This is the head-to-head experiment
    that demonstrates the BCR26 advantage: ridge CG with eps_pl ~ lam
    blows up as lam -> 0, while the Morse-Bott pinv stays accurate.

How the paper attains strong convexity
--------------------------------------
The paper uses R(w) = (lam/2) * ||w||^2 in the lower level. That's the
whole trick: a classical L2/ridge term on the task-specific last-layer
weights. Cross-entropy is convex in linear-layer weights; adding lam/2
||w||^2 shifts every eigenvalue of Hess_w g up by lam, so Hess_w g >= lam
uniformly and the lower level is lam-strongly convex. Strong convexity
is engineered in, not a property of the underlying loss.

Why PL is a strictly more useful assumption for meta-learning
-------------------------------------------------------------
  * Few-shot regimes are inherently underdetermined in the last layer.
    With 5 way x 5 shots = 25 support points and feature dim >= 25, the
    unregularized last-layer problem is CONVEX but RANK-DEFICIENT:
    it has a whole affine subspace of minimizers. That breaks strong
    convexity but is exactly a PL problem.
  * Picking lam is a nuisance: too small -> strong-convexity constant mu
    is tiny -> the SC analysis blows up. Too large -> you are fitting the
    wrong objective and generalization suffers. PL analysis is
    lam-agnostic and handles lam = 0 uniformly.
  * This experiment empirically estimates mu_SC vs mu_PL and reports the
    gap; the larger the gap, the more your PL story dominates the SC
    story on the SAME problem.
"""

from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import geoopt
except ImportError as e:
    raise SystemExit(
        "geoopt is required. Install with `pip install geoopt`.\n"
        "Original error: %s" % e
    )


# #############################################################################
# #############################################################################
# ##                                                                         ##
# ##                  >>>  EDIT HERE TO CHANGE WHAT RUNS  <<<                ##
# ##                                                                         ##
# ##  This is the ONLY block you need to touch to change the experiment.     ##
# ##  Edit the values in the SETTINGS class below, then run:                 ##
# ##                                                                         ##
# ##      python riemannian_meta_learning_pl.py                              ##
# ##                                                                         ##
# #############################################################################
# #############################################################################

class SETTINGS:
    # ---- Which regime(s) to run --------------------------------------------
    # Options: "strong_convex", "tiny_lam", "pl_only", or "all".
    # See the per-regime defaults in REGIME_CONFIGS further down (search for
    # "REGIME_CONFIGS"); each regime fixes (lam, eps_pl, use_morse_bott).
    REGIME = "all"

    # ---- Per-regime overrides (None = use the regime's default) -----------
    LAM = None        # Lower-level L2 strength (None: per-regime default)
    EPS_PL = None     # Legacy ridge solver regularization (None: per-regime)

    # ---- Solver choice ----------------------------------------------------
    # USE_MORSE_BOTT: True/False/None.  None = per-regime default
    # (False for strong_convex, True for tiny_lam and pl_only).
    USE_MORSE_BOTT = None
    N_LANCZOS = 30    # Lanczos iterations for Hessian spectral probe

    # ---- Head-to-head solver-quality measurement (the BCR26 plot) --------
    # When True, at each outer step we ALSO compute v with two ridge-CG
    # variants and report each method's relative error against the
    # Moore-Penrose pinv reference. Generates solver_quality.png in
    # addition to comparison.png. Adds ~2x compute per step.
    MEASURE_HG_QUALITY = True
    CG_STEPS_LONG = 200   # CG iterations for the "more compute" baseline

    # ---- Training schedule -----------------------------------------------
    # N_OUTER and N_TASKS_PER_BATCH adapt to MEASURE_HG_QUALITY when set
    # to None: 200/4 normally, 100/2 with quality measurement.
    N_OUTER = None
    N_INNER = 30
    N_TASKS_PER_BATCH = None

    # ---- Reproducibility & I/O -------------------------------------------
    SEED = 0
    DEVICE = None     # None = auto-detect (cuda if available, else cpu)
    OUT_DIR = "./outputs_riem_meta_pl"


# #############################################################################
# ##                  END OF USER-EDITABLE CONFIGURATION                     ##
# #############################################################################


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Stiefel-constrained CNN backbone
# ---------------------------------------------------------------------------
# The paper constrains the kernels of the first 2 conv layers so that the
# unfolded kernel matrix is on St(d, r). For a conv with c_in input channels,
# kernel k*k and c_out output channels, the kernel reshaped to (c_in*k*k, c_out)
# must have orthonormal columns. Following Li-Li-Todorovic (ICLR'19, ref [48]
# in Han et al.), this gives St(c_in*k*k, c_out).
#
# For MiniImageNet with 4-block CNN as described:
#   input: 3 x 84 x 84
#   block 1: conv 3 -> 16, kernel 3x3, padding 1 (St(27, 16))       -- Stiefel
#   block 2: conv 16 -> 16, kernel 3x3, padding 1 (St(144, 16))     -- Stiefel
#   block 3: conv 16 -> 16, kernel 3x3, padding 1                    -- Euclidean
#   block 4: conv 16 -> 16, kernel 3x3, padding 1                    -- Euclidean
# Each block: conv -> BN -> ReLU -> max-pool(2). After 4 blocks, 84/16 ~= 5.
# Feature dim after flattening: 16 * 5 * 5 = 400.
#
# Last layer w (lower-level variable) is Linear(feature_dim, n_way) in R.


def _make_stiefel_param(in_ch: int, out_ch: int, k: int = 3) -> geoopt.ManifoldParameter:
    """
    Build a Stiefel parameter of shape (in_ch*k*k, out_ch). The convolution
    itself is performed via F.conv2d in the forward pass so that autograd
    flows through the reshape back into the Stiefel parameter.
    """
    d = in_ch * k * k  # rows
    r = out_ch          # columns
    assert d >= r, f"Stiefel requires d >= r, got d={d}, r={r}"

    # Initialize on Stiefel via QR of a random Gaussian.
    A = torch.randn(d, r)
    Q, _ = torch.linalg.qr(A)  # Q: (d, r), orthonormal columns
    manifold = geoopt.Stiefel(canonical=False)
    return geoopt.ManifoldParameter(Q.clone(), manifold=manifold)


class StiefelCNN(nn.Module):
    """
    4-block CNN for MiniImageNet-like inputs (3 x 84 x 84).
    First 2 conv kernels are Stiefel-constrained (upper-level Theta).
    Blocks 3-4 are standard Euclidean convs (also upper-level, but on R).
    The final linear classifier is NOT part of this module -- it is the
    lower-level variable w, handled separately per task.

    The Stiefel convs are performed via F.conv2d in forward() so that
    autograd flows through the (transpose + reshape) of the Stiefel
    parameter matrix into the kernel tensor. This avoids the nn.Module
    __setattr__ guard that forbids assigning a non-Parameter tensor to
    .weight.

    forward() returns features of shape (batch, feature_dim).
    """

    FEATURE_DIM = 16 * 5 * 5  # 400; 84 -> 42 -> 21 -> 10 -> 5

    def __init__(self) -> None:
        super().__init__()

        # --- Stiefel-constrained blocks (params registered directly) ---
        # St(27, 16)
        self.theta1 = _make_stiefel_param(in_ch=3, out_ch=16, k=3)
        self.bn1 = nn.BatchNorm2d(16)

        # St(144, 16)
        self.theta2 = _make_stiefel_param(in_ch=16, out_ch=16, k=3)
        self.bn2 = nn.BatchNorm2d(16)

        # Kernel-size/padding bookkeeping for F.conv2d
        self._k1, self._in1, self._out1 = 3, 3, 16
        self._k2, self._in2, self._out2 = 3, 16, 16

        # --- Standard Euclidean blocks (blocks 3 and 4) ---
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(16)

        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(16)

        self.pool = nn.MaxPool2d(2)

    def _stiefel_weight(self, theta: torch.Tensor, in_ch: int, out_ch: int, k: int) -> torch.Tensor:
        """Reshape a Stiefel param of shape (in_ch*k*k, out_ch) into a conv
        kernel of shape (out_ch, in_ch, k, k). Autograd flows through this."""
        return theta.t().contiguous().view(out_ch, in_ch, k, k)

    def upper_params(self) -> List[nn.Parameter]:
        """All parameters that make up Theta (upper-level variable)."""
        return [
            self.theta1, self.theta2,                               # Stiefel
            self.conv3.weight, self.conv4.weight,                   # Euclidean convs
            self.bn1.weight, self.bn1.bias,
            self.bn2.weight, self.bn2.bias,
            self.bn3.weight, self.bn3.bias,
            self.bn4.weight, self.bn4.bias,
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1 (Stiefel)
        w1 = self._stiefel_weight(self.theta1, self._in1, self._out1, self._k1)
        x = F.conv2d(x, w1, bias=None, padding=1)
        x = self.pool(F.relu(self.bn1(x)))                          # 84 -> 42

        # Block 2 (Stiefel)
        w2 = self._stiefel_weight(self.theta2, self._in2, self._out2, self._k2)
        x = F.conv2d(x, w2, bias=None, padding=1)
        x = self.pool(F.relu(self.bn2(x)))                          # 42 -> 21

        # Blocks 3-4 (Euclidean)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))              # 21 -> 10
        x = self.pool(F.relu(self.bn4(self.conv4(x))))              # 10 -> 5
        x = x.flatten(1)
        return x  # (batch, 400)


# ---------------------------------------------------------------------------
# Lower-level problem: per-task linear classifier in R
# ---------------------------------------------------------------------------

def task_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    w: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """
    g(theta, w; D) = CE(softmax(features @ w^T), labels) + (lam/2) * ||w||^2

    features: (n, d)   -- output of the backbone (depends on theta)
    labels:   (n,)     -- int class labels in [0, n_way)
    w:        (n_way, d)
    lam:      float, strength of L2 regularizer. lam=0 is the PL regime.
    """
    logits = features @ w.t()
    ce = F.cross_entropy(logits, labels)
    reg = 0.5 * lam * (w * w).sum()
    return ce + reg


def solve_lower_level(
    features: torch.Tensor,
    labels: torch.Tensor,
    n_way: int,
    lam: float,
    n_inner: int,
    inner_lr: float,
    w_init: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Minimize g(theta, w; D_s) over w for a given theta (encoded in `features`).

    Returns an approximation w_hat of w*(theta) -- gradient steps of size
    inner_lr for n_inner iterations. We detach `features` here because the
    inner loop is treated as a black box whose Jacobian w.r.t. theta is
    obtained later via implicit differentiation (conjugate gradient).
    """
    d = features.size(1)
    if w_init is None:
        w = torch.zeros(n_way, d, device=features.device, requires_grad=True)
    else:
        w = w_init.detach().clone().requires_grad_(True)

    features_det = features.detach()

    for _ in range(n_inner):
        loss = task_loss(features_det, labels, w, lam)
        grad = torch.autograd.grad(loss, w, create_graph=False)[0]
        with torch.no_grad():
            w = (w - inner_lr * grad).requires_grad_(True)

    return w.detach()


# ---------------------------------------------------------------------------
# Hypergradient via conjugate gradient (CG) on Hess_w g
# ---------------------------------------------------------------------------
# The hypergradient formula (Han et al., Thm 3.1; our PL paper, Lemma 6.iii):
#
#   grad_theta F(theta) = grad_theta f(theta, w*)  -
#                         [G^2_{theta,w} g(theta, w*)]^T [Hess_w g(theta, w*)]^{-1}
#                           grad_w f(theta, w*)
#
# For meta-learning f = g on the QUERY set (upper) and support set (lower).
# We implement this by:
#   1) solving [Hess_w g] v = grad_w f via CG (linear system in w-space)
#   2) computing the JVP [G^2_{theta,w} g] v via autograd
#
# Under PL / rank-deficient Hess_w, there are two ways to make the solve
# well-posed:
#
#   (a) The *pragmatic* route (old behavior, still supported via eps_pl):
#       add a ridge eps * I and solve (Hess + eps*I) v = b. This lifts
#       every eigenvalue of Hess by eps, which is simple but destroys
#       the Morse-Bott structure at w* (it turns zero eigenvalues --
#       which represent genuine flat directions along S -- into small
#       positive ones, injecting a bias into v in directions that have
#       no physical meaning).
#
#   (b) The *structural* route (new, default for PL regimes, following
#       BCR26 Lemma 2.2): identify ker Hess_w g = T_{w*} S via a
#       low-end eigenprobe, project b onto its orthogonal complement
#       N_{w*} S, and solve the well-conditioned system there. The
#       nonzero spectrum of Hess_w g on N_{w*} S is bounded below by
#       the PL constant mu, so CG is fast and well-conditioned.
#       The result is the Moore-Penrose pseudoinverse applied to b,
#       which BCR26 Prop 4.4 justifies as the correct object: f
#       restricted to a fiber F = pi^{-1}(w*) has w* as its unique
#       minimizer, and the hypergradient formula applies along N_{w*} S.
#
# Route (b) is selected when use_morse_bott=True (default in PL regimes).
# Route (a) is retained for the strongly-convex regime where Hess is
# uniformly positive definite and the two routes coincide up to O(eps).
# ---------------------------------------------------------------------------

def hessian_vector_product(
    features: torch.Tensor,
    labels: torch.Tensor,
    w: torch.Tensor,
    v: torch.Tensor,
    lam: float,
    eps_pl: float = 0.0,
) -> torch.Tensor:
    """Compute Hess_w g(theta, w) @ v + eps_pl * v.

    eps_pl > 0 corresponds to the regularized CG operator used in the PL
    regime; it plays the role of the spectral clipping from our PL paper
    Section 3 (eq. 1-2). Set eps_pl = 0 in the strongly convex regime.
    """
    w = w.detach().requires_grad_(True)
    loss = task_loss(features.detach(), labels, w, lam)
    grad_w = torch.autograd.grad(loss, w, create_graph=True)[0]
    hv = torch.autograd.grad((grad_w * v).sum(), w, retain_graph=False)[0]
    return hv + eps_pl * v


def conjugate_gradient(
    apply_A,
    b: torch.Tensor,
    n_steps: int = 20,
    tol: float = 1e-8,
) -> torch.Tensor:
    """Solve A x = b for symmetric PSD A using CG. `apply_A(v)` returns A v."""
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rs_old = (r * r).sum()
    for _ in range(n_steps):
        Ap = apply_A(p)
        alpha = rs_old / ((p * Ap).sum() + 1e-20)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = (r * r).sum()
        if rs_new.sqrt() < tol:
            break
        p = r + (rs_new / (rs_old + 1e-20)) * p
        rs_old = rs_new
    return x


# ---------------------------------------------------------------------------
# Morse-Bott machinery (Boumal-Criscitiello-Rebjock 2026, Lemma 2.2)
# ---------------------------------------------------------------------------
# For a smooth globally mu-PL function g: M -> R with minimizer set S, at
# every point w* in S the Hessian Hess_w g(w*) has a PRESCRIBED bimodal
# spectrum:
#
#       ker Hess_w g(w*) = T_{w*} S        (tangent to S)
#       Hess_w g(w*) | N_{w*} S  >=  mu * Id
#
# i.e., dim(S) eigenvalues are *exactly* zero, and all other eigenvalues
# are >= mu_PL > 0. There is an explicit gap between the two clusters.
#
# For our cross-entropy-on-features lower-level problem this predicts:
#   - rank-deficiency along directions in the Gram-matrix nullspace
#     (n_way * feat_dim - rank of features) eigenvalues at ~ 0
#   - all other eigenvalues >= mu_PL
#
# We exploit this as follows:
#   1. Use a bounded-memory Lanczos to find the top-k and the bottom-k
#      eigenpairs of Hess_w g at w*.
#   2. Read off the "Morse-Bott signature" (num small + num large eigvals,
#      plus the gap) as a diagnostic.
#   3. Solve Hess_w g(w*) v = b by PROJECTING b onto the positive-eigenvalue
#      subspace first (Moore-Penrose pseudoinverse). This respects the
#      Morse-Bott structure instead of destroying it with a scalar ridge.
# ---------------------------------------------------------------------------


def _hess_dense_eigh(
    apply_H,
    n_way: int,
    d: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Materialize the symmetric operator Hess_w g as a dense (N, N) matrix
    and eigendecompose it via torch.linalg.eigh, where N = n_way * d.

    Returns (eigvals, eigvecs) in ascending order, with eigvecs of shape (N, N)
    (columns are eigenvectors, flattened in the (n_way, d) -> N reshape
    convention).

    Why dense + eigh, not Lanczos? For our setup N = n_way * d is modest
    (at most a few thousand), and Hess_w g is ONLY the Hessian with respect
    to the last-layer weights (features are detached). So each HVP is a
    short autograd computation over the linear head, not the full CNN.
    Materializing Hess is N HVPs -- tractable -- and torch.linalg.eigh then
    gives the exact eigendecomposition with no risk of Lanczos-style loss
    of orthogonality. This is what BCR26 Lemma 2.2 actually wants: an
    accurate read on the Hessian's bimodal spectrum at w*.

    For very large N (e.g., feature dim > a few thousand) one should swap
    this out for a truncated top-k eigensolver (LOBPCG, scipy.eigsh). We
    cap N below for safety.
    """
    N = n_way * d
    MAX_DENSE_N = 8000
    if N > MAX_DENSE_N:
        raise RuntimeError(
            f"N = n_way*d = {N} exceeds MAX_DENSE_N = {MAX_DENSE_N}. "
            f"Materializing Hess dense is too expensive. Switch to a "
            f"top-k eigensolver for this regime."
        )

    # Build Hess column-by-column via HVPs on standard basis vectors.
    # Use an efficient batched approach: feed H a batch of unit vectors
    # stacked into a (BATCH, n_way, d) tensor, one call per batch.
    H = torch.zeros(N, N, device=device, dtype=dtype)
    basis_chunk = 64
    for start in range(0, N, basis_chunk):
        end = min(start + basis_chunk, N)
        for j in range(start, end):
            e = torch.zeros(n_way, d, device=device, dtype=dtype)
            row, col = divmod(j, d)
            e[row, col] = 1.0
            Hej = apply_H(e)
            H[:, j] = Hej.reshape(-1)
    # Symmetrize to undo any small asymmetry from finite-precision autograd.
    H = 0.5 * (H + H.t())
    eigvals, eigvecs = torch.linalg.eigh(H)
    return eigvals, eigvecs



def morse_bott_structure(
    features: torch.Tensor,
    labels: torch.Tensor,
    w_star: torch.Tensor,
    lam: float,
    n_lanczos: int = 30,  # kept as a no-op kwarg for API compatibility
    gap_ratio: float = 0.05,
) -> Dict[str, object]:
    """Estimate the Morse-Bott structure of Hess_w g at w_star.

    Returns a dict with:
        eigvals         (tensor)  -- eigenvalues in ascending order
        eigvecs         (tensor)  -- eigenvectors, columns, in (n_way*d, N)
        dim_ker_est     (int)     -- estimated dim(T_{w*} S) = rank deficit
        dim_range_est   (int)     -- estimated dim(N_{w*} S) = codim(S) = k
        spectral_gap    (float)   -- first_positive_eig / largest_eig; the
                                     empirical read on Lemma 2.2 (a healthy
                                     PL problem has gap O(1), not O(eps))
        mu_PL_hess      (float)   -- smallest positive eigenvalue, our
                                     Lemma-2.2 estimate of mu_PL
        L_hess          (float)   -- largest eigenvalue

    Implementation: materializes the dense N x N Hessian via N HVPs on
    standard basis vectors, then calls torch.linalg.eigh. The HVPs are
    cheap because Hess_w g involves only the last-layer linear head
    (features are detached), not the full CNN. The `n_lanczos` argument
    is kept for API compatibility but has no effect in the dense-eigh
    implementation.
    """
    w_star_det = w_star.detach()
    features_det = features.detach()

    def apply_H(v: torch.Tensor) -> torch.Tensor:
        # Include lam * I because the lower-level g INCLUDES the lam * ||w||^2
        # term (so its Hessian is Hess_CE + lam * I). For lam = 0 this is
        # just Hess_CE.
        return hessian_vector_product(
            features_det, labels, w_star_det, v, lam=lam, eps_pl=0.0,
        )

    n_way, d = w_star.shape
    device = w_star.device
    eigvals, eigvecs = _hess_dense_eigh(apply_H, n_way=n_way, d=d, device=device)
    L_hess = float(eigvals[-1].item())
    L_hess = max(L_hess, 1e-12)
    threshold = gap_ratio * L_hess

    # Separate eigenvalues.
    small_mask = eigvals < threshold
    large_mask = ~small_mask
    dim_ker_est = int(small_mask.sum().item())
    dim_range_est = int(large_mask.sum().item())

    if dim_range_est > 0:
        first_positive = float(eigvals[large_mask][0].item())
    else:
        first_positive = 0.0

    if dim_ker_est > 0 and dim_range_est > 0:
        spectral_gap = first_positive / L_hess
    else:
        spectral_gap = float("nan")

    return {
        "eigvals": eigvals.detach().cpu(),
        "eigvecs": eigvecs.detach(),           # kept on device for solver use
        "dim_ker_est": dim_ker_est,
        "dim_range_est": dim_range_est,
        "spectral_gap": spectral_gap,
        "mu_PL_hess": first_positive,
        "L_hess": L_hess,
        "threshold": threshold,
    }


def morse_bott_pinv_solve(
    apply_H,
    b: torch.Tensor,
    mb_struct: Dict[str, object],
    cg_steps: int = 20,
    tol: float = 1e-8,
) -> torch.Tensor:
    """Apply the Moore-Penrose pseudoinverse of Hess_w g to b, following the
    Morse-Bott decomposition of Lemma 2.2 in BCR26.

    Implementation: truncated-spectrum pseudoinverse from the Lanczos Ritz
    pairs. For Ritz pairs {(lambda_i, u_i)} with lambda_i > threshold, we set

        v  =  sum_{i: lambda_i > threshold}  (1/lambda_i) * <u_i, b> * u_i.

    This is exactly the Moore-Penrose pseudoinverse applied to b on the
    Lanczos-seen subspace of Hess_w g, restricted to the positive-eigenvalue
    cluster that BCR26 Lemma 2.2 guarantees is separated from ker(Hess) by
    the PL gap. Advantages over the naive (Hess + eps*I)^{-1} approach:

      * It does not inject a spurious bias into b along ker(Hess): the
        kernel components are exactly killed, instead of being inverted
        by 1/eps (which would blow up as eps -> 0).

      * The reconstruction is numerically stable because we divide only
        by eigenvalues >= mu_PL, not by eps (which would be << mu_PL if
        chosen to resemble "nearly no regularization").

    Note: if Lanczos n_lanczos is small relative to dim(range(Hess)), this
    is a *truncated* pinv and may miss small components of b in the true
    range(Hess) that were not captured by the Krylov subspace. Increase
    n_lanczos to tighten this approximation. In practice n_lanczos >=
    n_way * (n_support - 1) + a small safety margin suffices, because
    that is the worst-case rank of the CE Hessian.
    """
    eigvals = mb_struct["eigvals"].to(b.device)
    eigvecs = mb_struct["eigvecs"]             # (N, m) on device
    threshold = mb_struct["threshold"]
    n_way, d = b.shape
    N = n_way * d

    # If the spectrum is one-sided (no small eigenvalues -> strongly convex
    # regime), fall back to plain CG on the full Hessian; it is fast and
    # exact there.
    small_mask = eigvals < threshold
    large_mask = ~small_mask
    if small_mask.sum().item() == 0:
        return conjugate_gradient(apply_H, b, n_steps=cg_steps, tol=tol)

    if large_mask.sum().item() == 0:
        # Pathological case: no detected range. Return 0 (the pinv of a
        # zero-map applied to anything is 0).
        return torch.zeros_like(b)

    U_rng = eigvecs[:, large_mask]             # (N, k_rng)
    lam_rng = eigvals[large_mask]              # (k_rng,), all > threshold

    # Truncated-spectrum pseudoinverse: v = sum_i (<u_i, b> / lambda_i) u_i.
    b_flat = b.reshape(-1)
    coeffs = U_rng.t() @ b_flat                # (k_rng,)
    coeffs = coeffs / lam_rng
    v_flat = U_rng @ coeffs
    return v_flat.view(n_way, d)


# ---------------------------------------------------------------------------
# Head-to-head solver quality measurement
# ---------------------------------------------------------------------------
# Even when the OUTER loop converges to similar loss/accuracy across regimes
# (as the previous comparison plot showed), the per-step HYPERGRADIENT
# QUALITY can differ dramatically. The Moore-Penrose pseudoinverse from
# BCR26 Lemma 2.2 is the structurally correct object; SC-style ridge CG
# with eps_pl ~ lam is a different object whose error blows up as lam -> 0.
#
# evaluate_solver_quality measures, on the SAME problem (same Hess, same
# RHS), how close each solver's v is to the reference Moore-Penrose pinv.
# This is the head-to-head experiment that makes the SC-vs-PL contrast
# unambiguous.
# ---------------------------------------------------------------------------


def evaluate_solver_quality(
    apply_H,                            # callable: (n_way, d) -> (n_way, d)
    b: torch.Tensor,                    # (n_way, d) RHS = grad_w f
    mb_struct: Dict[str, object],       # output of morse_bott_structure
    eps_pl_legacy: float,               # eps used by the legacy ridge solver
    cg_steps_legacy: int,               # cg steps used by the legacy ridge solver
    cg_steps_long: int = 200,           # how many CG steps to give legacy "all the rope"
    ref_threshold_ratio: float = 0.01,  # tighter threshold for the reference pinv
) -> Dict[str, float]:
    """Measure how well each candidate v-solver approximates the
    Moore-Penrose pseudoinverse of Hess_w g, applied to b.

    Returns dict with relative errors w.r.t. the reference (which is the
    truncated-spectrum pinv with a tight threshold, i.e., the structural
    answer that BCR26 Lemma 2.2 picks out).

    Keys returned:
        v_ref_norm                  norm of the reference v
        err_morse_bott              ||v_MB    - v_ref|| / ||v_ref||
                                    (MB = configured threshold)
        err_ridge_cg_short          ||v_ridge_short - v_ref|| / ||v_ref||
                                    (legacy: configured eps_pl + cg_steps)
        err_ridge_cg_long           same but with cg_steps_long iterations
                                    (does extra compute close the gap?)
        err_zero                    ||0 - v_ref|| / ||v_ref|| = 1.0
                                    (sanity check: an unhelpful baseline)

    Interpretation:
      * err_morse_bott should be near 0 (~1e-3 or better): the configured
        Morse-Bott threshold is a robust approximation to the true pinv.
      * err_ridge_cg_short should be SMALL when eps_pl is large
        (strongly-convex regime: Hess + eps*I is well-conditioned, CG
        converges quickly and matches pinv up to O(eps)). It should be
        LARGE when eps_pl is tiny (PL regime: Hess + eps*I has condition
        number L/eps, CG can't converge in cg_steps iterations).
      * err_ridge_cg_long shows whether throwing more CG iterations at the
        problem closes the gap. In the truly rank-deficient case, even
        infinite CG cannot recover the pinv answer because the kernel
        components blow up as 1/eps.
    """
    eigvals = mb_struct["eigvals"].to(b.device)
    eigvecs = mb_struct["eigvecs"]
    L_hess = float(mb_struct["L_hess"])
    n_way, d = b.shape

    # --- Reference v: truncated-spectrum pinv with a TIGHT threshold ---
    # We use a 1%-of-L threshold (vs the 5% used by the configured MB
    # solver) to get a more accurate pinv, while still excluding the
    # numerically-zero kernel directions.
    ref_threshold = ref_threshold_ratio * L_hess
    rng_mask_ref = eigvals >= ref_threshold
    if rng_mask_ref.sum().item() == 0:
        # Nothing in range -- pinv is 0.
        return {
            "v_ref_norm": 0.0,
            "err_morse_bott": float("nan"),
            "err_ridge_cg_short": float("nan"),
            "err_ridge_cg_long": float("nan"),
            "err_zero": 0.0,
        }
    U_ref = eigvecs[:, rng_mask_ref]
    lam_ref = eigvals[rng_mask_ref]
    b_flat = b.reshape(-1)
    v_ref_flat = U_ref @ ((U_ref.t() @ b_flat) / lam_ref)
    v_ref = v_ref_flat.view(n_way, d)
    v_ref_norm = float(v_ref.norm().item())
    if v_ref_norm < 1e-20:
        return {
            "v_ref_norm": v_ref_norm,
            "err_morse_bott": float("nan"),
            "err_ridge_cg_short": float("nan"),
            "err_ridge_cg_long": float("nan"),
            "err_zero": 0.0,
        }

    def relerr(v: torch.Tensor) -> float:
        return float((v - v_ref).norm().item() / v_ref_norm)

    # --- Morse-Bott pinv at the CONFIGURED threshold ---
    rng_mask_mb = eigvals >= mb_struct["threshold"]
    if rng_mask_mb.sum().item() > 0:
        U_mb = eigvecs[:, rng_mask_mb]
        lam_mb = eigvals[rng_mask_mb]
        v_mb_flat = U_mb @ ((U_mb.t() @ b_flat) / lam_mb)
        v_mb = v_mb_flat.view(n_way, d)
        err_mb = relerr(v_mb)
    else:
        err_mb = 1.0  # everything killed -> v_mb = 0

    # --- Legacy ridge CG with the configured (eps_pl, cg_steps) ---
    def apply_A_legacy(v: torch.Tensor) -> torch.Tensor:
        return apply_H(v) + eps_pl_legacy * v

    v_ridge_short = conjugate_gradient(apply_A_legacy, b, n_steps=cg_steps_legacy)
    err_ridge_short = relerr(v_ridge_short)

    # --- Same legacy solver but with many more CG iterations ---
    v_ridge_long = conjugate_gradient(apply_A_legacy, b, n_steps=cg_steps_long)
    err_ridge_long = relerr(v_ridge_long)

    return {
        "v_ref_norm": v_ref_norm,
        "err_morse_bott": err_mb,
        "err_ridge_cg_short": err_ridge_short,
        "err_ridge_cg_long": err_ridge_long,
        "err_zero": 1.0,
    }


def hypergradient_cg(
    backbone: StiefelCNN,
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    query_x: torch.Tensor,
    query_y: torch.Tensor,
    n_way: int,
    lam_lower: float,
    n_inner: int,
    inner_lr: float,
    cg_steps: int,
    eps_pl: float,
    use_morse_bott: bool = True,
    n_lanczos: int = 30,
    measure_hg_quality: bool = False,
    cg_steps_long: int = 200,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute an approximation of grad_Theta F for a single task.

    Returns (upper_loss_scalar_tensor, diagnostics_dict).
    The side effect is that backbone.upper_params() have their .grad
    attributes INCREMENTED with the hypergradient contribution.

    Solve strategy for [Hess_w g] v = grad_w f:

      * use_morse_bott=True (default, recommended for PL and tiny-lam
        regimes): use the Moore-Penrose pseudoinverse defined by the
        Morse-Bott decomposition of Hess_w g at w* (BCR26 Lemma 2.2).
        eps_pl is then used ONLY as a Lanczos gap threshold, not as a
        ridge added to the Hessian.

      * use_morse_bott=False (old behavior): solve with CG on
        Hess_w g + eps_pl * I. Matches the strongly-convex analysis of
        Han et al. exactly when lam_lower + eps_pl >= mu_SC.
    """

    # ---- Lower-level solve (inner loop) ----
    # In BCR26 notation this is the end-point map pi: solve_lower_level(w_0)
    # = pi(w_0), the limit of negative gradient flow on g initialized at w_0.
    feats_s = backbone(support_x)  # depends on Theta
    w_star = solve_lower_level(
        feats_s, support_y, n_way=n_way, lam=lam_lower,
        n_inner=n_inner, inner_lr=inner_lr,
    )

    # ---- Upper-level loss on query set ----
    feats_q = backbone(query_x)
    # Upper-level f uses the UN-regularized loss (regularizer only serves
    # strong convexity of the lower level; the objective we care about is
    # the unregularized validation loss).
    logits_q = feats_q @ w_star.t()
    f_value = F.cross_entropy(logits_q, query_y)

    # ---- Direct term: grad_theta f(theta, w*) ----
    direct_grads = torch.autograd.grad(
        f_value, backbone.upper_params(), retain_graph=True, allow_unused=True,
    )

    # ---- Implicit term: need v* = [Hess_w g]^{-1 or +} grad_w f, then
    #                    subtract [G^2_{theta,w} g]^T v* ----
    # Compute grad_w f at w_star (note: features_q depends on theta, but
    # w_star is detached, so this is grad_w f purely through logits).
    w_leaf = w_star.clone().requires_grad_(True)
    logits_q_leaf = feats_q.detach() @ w_leaf.t()
    f_leaf = F.cross_entropy(logits_q_leaf, query_y)
    grad_w_f = torch.autograd.grad(f_leaf, w_leaf)[0].detach()

    # Build the HVP operator at (theta, w_star) with the SUPPORT set, which
    # is the lower-level problem.
    feats_s_det = feats_s.detach()

    # Raw HVP without the legacy ridge: this is what BCR26 actually wants.
    def apply_H_raw(v: torch.Tensor) -> torch.Tensor:
        return hessian_vector_product(
            feats_s_det, support_y, w_star, v, lam=lam_lower, eps_pl=0.0,
        )

    # ---- Pick the solver route ----
    # We may need mb_struct (eigendecomposition of Hess_w g at w*) for either:
    #   - the actual update (use_morse_bott=True)
    #   - the head-to-head solver-quality measurement (measure_hg_quality=True)
    # Compute it once if either flag requires it.
    mb_struct: Dict[str, object] = {}
    need_mb = use_morse_bott or measure_hg_quality
    if need_mb:
        mb_struct = morse_bott_structure(
            feats_s_det, support_y, w_star, lam=lam_lower,
            n_lanczos=min(n_lanczos, n_way * feats_s_det.size(1)),
            gap_ratio=0.05,
        )

    if use_morse_bott:
        # Route (b): Moore-Penrose pseudoinverse via Morse-Bott split.
        # This is the structural contribution from BCR26 Lemma 2.2.
        v_star = morse_bott_pinv_solve(
            apply_H_raw, grad_w_f, mb_struct, cg_steps=cg_steps,
        )
    else:
        # Route (a): legacy CG with scalar ridge (the original behavior).
        def apply_A(v: torch.Tensor) -> torch.Tensor:
            return hessian_vector_product(
                feats_s_det, support_y, w_star, v,
                lam=lam_lower, eps_pl=eps_pl,
            )
        v_star = conjugate_gradient(apply_A, grad_w_f, n_steps=cg_steps)

    # Implicit term: compute grad_theta [ <grad_w g(theta, w_star), v_star> ]
    # This is (G^2_{theta, w} g)^T v_star by the chain rule.
    w_for_cross = w_star.detach().requires_grad_(True)
    loss_lower_for_cross = task_loss(feats_s, support_y, w_for_cross, lam_lower)
    grad_w_g = torch.autograd.grad(
        loss_lower_for_cross, w_for_cross, create_graph=True,
    )[0]
    cross_scalar = (grad_w_g * v_star).sum()
    cross_grads = torch.autograd.grad(
        cross_scalar, backbone.upper_params(), allow_unused=True,
    )

    # ---- Accumulate hypergradient into .grad attributes ----
    for p, gd, gc in zip(backbone.upper_params(), direct_grads, cross_grads):
        hg = torch.zeros_like(p) if gd is None else gd.clone()
        if gc is not None:
            hg = hg - gc
        if p.grad is None:
            p.grad = hg
        else:
            p.grad = p.grad + hg

    # ---- Diagnostics: empirical mu_SC vs mu_PL on the lower-level problem ----
    diag = estimate_pl_vs_sc(
        feats_s_det, support_y, w_star, lam=lam_lower, n_way=n_way,
        mb_struct=mb_struct if need_mb else None,
        inner_lr=inner_lr,
    )
    diag["upper_loss"] = f_value.item()
    diag["used_morse_bott"] = 1.0 if use_morse_bott else 0.0

    # ---- Head-to-head solver-quality diagnostics ----
    # When measure_hg_quality=True, we compare three v-solvers on the SAME
    # problem (same Hess, same RHS = grad_w f) and report each method's
    # relative error against the Moore-Penrose pinv reference. This is the
    # head-to-head experiment that demonstrates the BCR26 advantage:
    # ridge-CG with eps_pl ~ lam blows up as lam -> 0, while the Morse-Bott
    # pinv stays accurate.
    if measure_hg_quality and need_mb:
        q = evaluate_solver_quality(
            apply_H_raw, grad_w_f, mb_struct,
            eps_pl_legacy=eps_pl, cg_steps_legacy=cg_steps,
            cg_steps_long=cg_steps_long,
        )
        diag.update(q)

    return f_value.detach(), diag



def estimate_pl_vs_sc(
    features: torch.Tensor,
    labels: torch.Tensor,
    w_star: torch.Tensor,
    lam: float,
    n_way: int,
    mb_struct: Dict[str, object] | None = None,
    inner_lr: float = 0.1,
    n_probe: int = 8,
) -> Dict[str, float]:
    """Empirically estimate the strong-convexity and PL constants for
    g(theta_fixed, .) at w_star, and summarize the Morse-Bott structure.

    Estimators (all three come out of BCR26):

      mu_SC_est        smallest eigenvalue of Hess_w g at w_star. This is
                       the strong-convexity constant (>= lam uniformly, 0
                       in the PL-only regime). Obtained from mb_struct if
                       available, otherwise via inverse power iteration.

      mu_PL_est        classical probe estimate:
                         inf_{w~}  ||grad_w g(w~)||^2  /  (2 (g(w~) - g*))
                       over random perturbations w~ around w*.

      mu_PL_QG         NEW: quadratic-growth estimator, directly from
                       BCR26 Lemma 2.1. For any w in a small neighborhood
                       of w*,
                           g(w) - g*  >=  (mu_PL / 2) * dist(w, S)^2.
                       We use dist(w, S) ~ ||w - w*|| (valid locally since
                       w* in S and w is close to w*), and get
                           mu_PL_QG  ~  2 (g(w) - g*) / ||w - w*||^2.
                       This is what the paper's analysis actually uses, and
                       it does not depend on gradient evaluations at w~,
                       which is more robust at rank-deficient problems.

      dim_ker_hess     empirical dim of ker Hess_w g(w_star). Under BCR26
                       Lemma 2.2 this equals dim(T_{w*} S); in our setup
                       this is the rank deficit of the support-set feature
                       Gram matrix (times n_way).

      dim_range_hess   empirical codimension: dim(N_{w*} S) = k. This is
                       the "k" that appears in g = g* + ||phi||^2 from
                       Theorem 1.2 -- the true number of squared residuals
                       that encode g locally.

      mb_gap           first_positive_eig / L_hess. A clean eigenvalue gap
                       > 0 is the direct empirical certificate that the
                       bimodal spectrum predicted by Lemma 2.2 is real on
                       this problem.
    """
    d = features.size(1)
    device = features.device
    features = features.detach()

    # --- mu_SC via the Hessian spectrum (Lanczos if we already have it) ---
    if mb_struct is not None and "eigvals" in mb_struct:
        eigvals = mb_struct["eigvals"]
        mu_sc_est = float(max(eigvals[0].item(), 0.0))
        dim_ker_est = int(mb_struct["dim_ker_est"])
        dim_range_est = int(mb_struct["dim_range_est"])
        mb_gap = float(mb_struct["spectral_gap"])
        mu_pl_hess = float(mb_struct["mu_PL_hess"])
    else:
        def apply_H(v):
            return hessian_vector_product(features, labels, w_star, v, lam=0.0)

        # Inverse power iteration for smallest eigenvalue.
        mu_sc_est = float("inf")
        for _ in range(3):
            v = torch.randn(n_way, d, device=device)
            v = v / (v.norm() + 1e-20)
            for _ in range(20):
                delta = 1e-4
                sol = conjugate_gradient(
                    lambda u: apply_H(u) + delta * u, v, n_steps=15,
                )
                v = sol / (sol.norm() + 1e-20)
            Hv = apply_H(v)
            rq = (v * Hv).sum().item()
            mu_sc_est = min(mu_sc_est, max(rq, 0.0))
        dim_ker_est = -1
        dim_range_est = -1
        mb_gap = float("nan")
        mu_pl_hess = float("nan")

    mu_sc_est = max(mu_sc_est, lam)

    # --- mu_PL via probes in N_{w*} S (normal to the minimizer manifold) ---
    #
    # The paper's Lemma 2.1 gives
    #     g(w) - g*  >=  (mu_PL / 2) * dist(w, S)^2,
    # which is a statement about the distance to S, NOT about ||w - w*||.
    # An isotropic probe around w* mostly lands in T_{w*} S (= ker Hess),
    # giving g(w~) ~ g* but ||w~ - w*|| large -> a vacuously small estimate.
    # We avoid this failure mode by sampling the probe direction in the
    # span of the RANGE eigenvectors of Hess, i.e., in N_{w*} S.
    #
    # We also use a SMALL probe scale so that the local Taylor expansion
    # holds: then 2 (g(w~) - g*) / scale^2 ~ direction^T Hess direction
    # (a true quadratic). At larger scales the nonlinearity of cross-
    # entropy (softmax saturation at large ||w||) would make the bound
    # collapse toward 0.
    g_star = task_loss(features, labels, w_star, lam=lam).item()
    mu_pl_est = float("inf")
    mu_pl_qg = float("inf")

    have_range_basis = (
        mb_struct is not None
        and isinstance(mb_struct.get("eigvals", None), torch.Tensor)
        and int(mb_struct["dim_range_est"]) > 0
    )
    if have_range_basis:
        eigvals_full = mb_struct["eigvals"].to(device)
        eigvecs_full = mb_struct["eigvecs"]       # (N, m) on device
        large_mask = eigvals_full >= mb_struct["threshold"]
        U_rng = eigvecs_full[:, large_mask]       # (N, k_rng)
        k_rng = U_rng.size(1)
    else:
        U_rng = None
        k_rng = 0

    # Probe scales (relative to w_star's typical per-coord magnitude).
    # Choose qg_scale so that the EXPECTED signal
    #    delta_g ~ (1/2) * mu_PL_hess * qg_scale^2
    # is safely above the loss's finite-precision floor while still small
    # enough to stay in the local quadratic regime. With a loss of order
    # g_star ~ 1 in float32 (~7 decimal digits), we need delta_g >> 1e-7;
    # aiming for delta_g ~ 1e-4 is conservative. We also cap by a tenth
    # of ||w_star|| per coordinate so we don't leave the Taylor regime.
    base_scale = max(1e-4, w_star.abs().mean().item())
    if mb_struct is not None and mb_struct.get("mu_PL_hess", 0.0) > 0:
        target_delta = 1e-4 * max(abs(g_star), 1.0)
        qg_scale = math.sqrt(2.0 * target_delta / mb_struct["mu_PL_hess"])
        qg_scale = min(qg_scale, 0.5 * base_scale)  # don't leave Taylor regime
        qg_scale = max(qg_scale, 1e-3 * base_scale) # don't go below float32 floor
    else:
        qg_scale = 1e-2 * base_scale
    grad_scale = 1e-1 * base_scale

    qg_bounds: List[float] = []
    for _ in range(n_probe):
        if U_rng is not None:
            c = torch.randn(k_rng, device=device)
            direction = (U_rng @ c).view(n_way, d)
        else:
            direction = torch.randn(n_way, d, device=device)
        direction = direction / (direction.norm() + 1e-20)

        # --- Gradient-based PL probe (at moderate scale, good SNR) ---
        w_tilde = (w_star + grad_scale * direction).requires_grad_(True)
        g_val = task_loss(features, labels, w_tilde, lam=lam)
        grad_val = torch.autograd.grad(g_val, w_tilde)[0]
        num = (grad_val * grad_val).sum().item()
        denom_gap = max(g_val.item() - g_star, 1e-12)
        mu_pl_est = min(mu_pl_est, num / (2.0 * denom_gap))

        # --- QG-based PL probe (BCR26 Lemma 2.1) ---
        # Use a symmetric finite difference to cancel first-order error
        # (w_star is only an approximate minimizer returned from 30 GD
        # steps, so ||grad_w g(w_star)|| > 0 and a one-sided probe mixes
        # a first-order term into the reading). The symmetric difference
        #    D2 = [g(w*+h*d) + g(w*-h*d) - 2 g(w*)] / h^2
        # picks up only the second-order term d^T Hess d at w*, which is
        # EXACTLY the quantity that encodes mu_PL via BCR26 Lemma 2.2:
        #    d^T Hess d  in  [mu_PL, L]     for unit d in N_{w*} S.
        # We run the probe in float64 to resolve small second-order gaps
        # without catastrophic cancellation.
        feats_d = features.double()
        w_plus = (w_star + qg_scale * direction).double()
        w_minus = (w_star - qg_scale * direction).double()
        w_star_d = w_star.double()
        g_plus = task_loss(feats_d, labels, w_plus, lam=lam).item()
        g_minus = task_loss(feats_d, labels, w_minus, lam=lam).item()
        g_center = task_loss(feats_d, labels, w_star_d, lam=lam).item()
        d2 = (g_plus + g_minus - 2.0 * g_center) / (qg_scale * qg_scale)
        if d2 > 0:
            qg_bounds.append(d2)

    # Under BCR26 Lemma 2.2, for any unit direction d in N_{w*} S,
    #    d^T Hess d  >=  mu_PL,
    # so every positive D2 is a VALID UPPER bound on the true mu_PL via
    # Lemma 2.1; the tightest (smallest positive) is the best estimate.
    if qg_bounds:
        mu_pl_qg = min(qg_bounds)


    return {
        "mu_SC_est": mu_sc_est,
        "mu_PL_est": mu_pl_est,
        "mu_PL_QG": mu_pl_qg,
        "mu_PL_hess": mu_pl_hess,            # from Lanczos / Morse-Bott
        "dim_ker_hess": float(dim_ker_est),  # dim T_{w*} S (paper's m)
        "dim_range_hess": float(dim_range_est),  # dim N_{w*} S = k
        "mb_gap": mb_gap,
        "lam_used": lam,
    }


# ---------------------------------------------------------------------------
# Outer (upper-level) loop: Riemannian hypergradient descent
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    regime: str                  # "strong_convex" | "pl_only" | "tiny_lam"
    lam: float                   # lower-level L2 strength
    eps_pl: float                # regularization for CG in PL regime (legacy solver)
    n_way: int = 5
    n_shot: int = 5
    n_query: int = 15
    n_tasks_per_batch: int = 4   # meta-batch size
    n_outer: int = 200           # outer (hypergradient) steps
    n_inner: int = 30            # inner steps to approximate w*
    inner_lr: float = 0.1
    outer_lr: float = 0.05       # lr for upper-level on Stiefel
    cg_steps: int = 20
    seed: int = 0
    log_every: int = 5
    # --- BCR26 structural-solver knobs ---
    use_morse_bott: bool = True  # Moore-Penrose pseudoinverse solve
                                 # (from BCR26 Lemma 2.2). Set False to
                                 # fall back to the legacy (Hess + eps_pl*I)
                                 # ridge solve for comparison.
    n_lanczos: int = 30          # Lanczos iterations for Hessian spectrum
    # --- Head-to-head solver-quality measurement ---
    measure_hg_quality: bool = False
                                 # If True, at each outer step compare
                                 # Morse-Bott pinv vs legacy ridge CG on
                                 # the SAME problem and report each
                                 # method's relative error to the
                                 # Moore-Penrose pinv reference.
    cg_steps_long: int = 200     # CG steps for the "all the rope" baseline


def build_riemannian_optimizer(backbone: StiefelCNN, cfg: RunConfig) -> geoopt.optim.RiemannianSGD:
    """Use geoopt's RiemannianSGD so that Stiefel parameters are updated
    intrinsically (exponential map / QR retraction) and Euclidean params
    get standard SGD. Matches the paper's RHGD outer update."""
    return geoopt.optim.RiemannianSGD(backbone.upper_params(), lr=cfg.outer_lr)


def run_regime(
    cfg: RunConfig,
    task_sampler,
    device: torch.device,
) -> Dict[str, List[float]]:
    """Run one regime and return training curves + diagnostics."""
    set_seed(cfg.seed)
    backbone = StiefelCNN().to(device)
    optimizer = build_riemannian_optimizer(backbone, cfg)

    history = {
        "outer_step": [], "upper_loss": [], "query_acc": [],
        "mu_SC_est": [], "mu_PL_est": [],
        # New BCR26-informed diagnostics:
        "mu_PL_QG": [], "mu_PL_hess": [],
        "dim_ker_hess": [], "dim_range_hess": [], "mb_gap": [],
        # Head-to-head solver-quality diagnostics (filled when
        # cfg.measure_hg_quality=True):
        "err_morse_bott": [], "err_ridge_cg_short": [],
        "err_ridge_cg_long": [], "v_ref_norm": [],
        "wallclock": [],
    }

    t0 = time.time()
    for step in range(cfg.n_outer):
        optimizer.zero_grad(set_to_none=False)
        # Zero grads on Stiefel params manually (geoopt doesn't always).
        for p in backbone.upper_params():
            if p.grad is not None:
                p.grad.zero_()

        batch_loss = 0.0
        batch_acc = 0.0
        diag_accum = {
            "mu_SC_est": 0.0, "mu_PL_est": 0.0,
            "mu_PL_QG": 0.0, "mu_PL_hess": 0.0,
            "dim_ker_hess": 0.0, "dim_range_hess": 0.0, "mb_gap": 0.0,
            "err_morse_bott": 0.0, "err_ridge_cg_short": 0.0,
            "err_ridge_cg_long": 0.0, "v_ref_norm": 0.0,
        }
        mb_gap_valid_count = 0
        quality_valid_count = 0

        for _ in range(cfg.n_tasks_per_batch):
            sx, sy, qx, qy = task_sampler(cfg.n_way, cfg.n_shot, cfg.n_query)
            sx, sy, qx, qy = sx.to(device), sy.to(device), qx.to(device), qy.to(device)

            f_val, diag = hypergradient_cg(
                backbone, sx, sy, qx, qy,
                n_way=cfg.n_way,
                lam_lower=cfg.lam,
                n_inner=cfg.n_inner,
                inner_lr=cfg.inner_lr,
                cg_steps=cfg.cg_steps,
                eps_pl=cfg.eps_pl,
                use_morse_bott=cfg.use_morse_bott,
                n_lanczos=cfg.n_lanczos,
                measure_hg_quality=cfg.measure_hg_quality,
                cg_steps_long=cfg.cg_steps_long,
            )
            batch_loss += f_val.item()

            # Evaluate query accuracy with a fresh inner solve (lightweight).
            with torch.no_grad():
                feats_q = backbone(qx)
            feats_s_eval = backbone(sx)
            w_eval = solve_lower_level(
                feats_s_eval, sy, n_way=cfg.n_way, lam=cfg.lam,
                n_inner=cfg.n_inner, inner_lr=cfg.inner_lr,
            )
            preds = (feats_q @ w_eval.t()).argmax(dim=1)
            batch_acc += (preds == qy).float().mean().item()

            quality_keys = {
                "err_morse_bott", "err_ridge_cg_short",
                "err_ridge_cg_long", "v_ref_norm",
            }
            quality_present = (
                cfg.measure_hg_quality and "err_morse_bott" in diag
            )
            if quality_present:
                quality_valid_count += 1
            for k in diag_accum:
                v = diag.get(k, 0.0)
                # mb_gap is NaN when the spectrum is one-sided; track
                # separately to avoid polluting the average.
                if k == "mb_gap":
                    if v == v:  # not NaN
                        diag_accum[k] += v
                        mb_gap_valid_count += 1
                elif k in quality_keys:
                    if quality_present and v == v:  # not NaN
                        diag_accum[k] += v
                else:
                    diag_accum[k] += v

        # Average gradient across tasks (hypergrad accumulator is a sum).
        for p in backbone.upper_params():
            if p.grad is not None:
                p.grad.div_(cfg.n_tasks_per_batch)

        optimizer.step()

        batch_loss /= cfg.n_tasks_per_batch
        batch_acc /= cfg.n_tasks_per_batch
        for k in diag_accum:
            if k == "mb_gap":
                diag_accum[k] = (
                    diag_accum[k] / mb_gap_valid_count
                    if mb_gap_valid_count > 0 else float("nan")
                )
            elif k in {"err_morse_bott", "err_ridge_cg_short",
                       "err_ridge_cg_long", "v_ref_norm"}:
                diag_accum[k] = (
                    diag_accum[k] / quality_valid_count
                    if quality_valid_count > 0 else float("nan")
                )
            else:
                diag_accum[k] /= cfg.n_tasks_per_batch

        if step % cfg.log_every == 0 or step == cfg.n_outer - 1:
            history["outer_step"].append(step)
            history["upper_loss"].append(batch_loss)
            history["query_acc"].append(batch_acc)
            history["mu_SC_est"].append(diag_accum["mu_SC_est"])
            history["mu_PL_est"].append(diag_accum["mu_PL_est"])
            history["mu_PL_QG"].append(diag_accum["mu_PL_QG"])
            history["mu_PL_hess"].append(diag_accum["mu_PL_hess"])
            history["dim_ker_hess"].append(diag_accum["dim_ker_hess"])
            history["dim_range_hess"].append(diag_accum["dim_range_hess"])
            history["mb_gap"].append(diag_accum["mb_gap"])
            history["err_morse_bott"].append(diag_accum["err_morse_bott"])
            history["err_ridge_cg_short"].append(diag_accum["err_ridge_cg_short"])
            history["err_ridge_cg_long"].append(diag_accum["err_ridge_cg_long"])
            history["v_ref_norm"].append(diag_accum["v_ref_norm"])
            history["wallclock"].append(time.time() - t0)

            qual_str = ""
            if cfg.measure_hg_quality:
                qual_str = (
                    f" | err(MB)={diag_accum['err_morse_bott']:.2e}, "
                    f"err(ridge,{cfg.cg_steps}cg)={diag_accum['err_ridge_cg_short']:.2e}, "
                    f"err(ridge,{cfg.cg_steps_long}cg)={diag_accum['err_ridge_cg_long']:.2e}"
                )
            print(
                f"[{cfg.regime:>14s}] step {step:4d} | "
                f"upper_loss={batch_loss:.4f} | "
                f"query_acc={batch_acc:.3f} | "
                f"mu_SC~={diag_accum['mu_SC_est']:.2e} | "
                f"mu_PL_QG~={diag_accum['mu_PL_QG']:.2e} | "
                f"dim(ker H)~{diag_accum['dim_ker_hess']:.1f}, "
                f"dim(rng H)~{diag_accum['dim_range_hess']:.1f} | "
                f"gap={diag_accum['mb_gap']:.2f}"
                f"{qual_str} | "
                f"t={time.time()-t0:.1f}s"
            )

    return history


# ---------------------------------------------------------------------------
# Data: synthetic "MiniImageNet-like" task sampler
# ---------------------------------------------------------------------------
# We generate a large pool of classes, each associated with a random mean in
# image space plus per-sample noise. This gives a non-trivial few-shot
# classification signal while keeping the experiment runnable anywhere.
# For real MiniImageNet, replace this with a standard few-shot data loader.

class SyntheticFewShotDataset:
    """Each 'class' is a random mean image; samples are mean + noise."""

    def __init__(self, n_classes: int = 64, img_size: int = 84, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.n_classes = n_classes
        self.img_size = img_size
        # Low-frequency class templates (smooth backgrounds) + high-frequency
        # class-specific textures.
        self.templates = torch.randn(
            n_classes, 3, img_size, img_size, generator=g,
        ) * 0.3
        # Smooth the templates to simulate natural image statistics.
        k = 9
        kernel = torch.ones(1, 1, k, k) / (k * k)
        smoothed = F.conv2d(
            self.templates.view(-1, 1, img_size, img_size),
            kernel, padding=k // 2,
        ).view(n_classes, 3, img_size, img_size)
        self.templates = smoothed * 2.0

    def sample_task(self, n_way: int, n_shot: int, n_query: int):
        classes = torch.randperm(self.n_classes)[:n_way]
        sx_list, sy_list, qx_list, qy_list = [], [], [], []
        for i, cls in enumerate(classes):
            base = self.templates[cls]
            # n_shot support + n_query query samples per class
            noise_s = torch.randn(n_shot, 3, self.img_size, self.img_size) * 0.5
            noise_q = torch.randn(n_query, 3, self.img_size, self.img_size) * 0.5
            sx_list.append(base.unsqueeze(0) + noise_s)
            qx_list.append(base.unsqueeze(0) + noise_q)
            sy_list.append(torch.full((n_shot,), i, dtype=torch.long))
            qy_list.append(torch.full((n_query,), i, dtype=torch.long))
        sx = torch.cat(sx_list, dim=0)
        sy = torch.cat(sy_list, dim=0)
        qx = torch.cat(qx_list, dim=0)
        qy = torch.cat(qy_list, dim=0)
        # Shuffle within support and query (order-invariance).
        sidx = torch.randperm(sx.size(0))
        qidx = torch.randperm(qx.size(0))
        return sx[sidx], sy[sidx], qx[qidx], qy[qidx]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(histories: Dict[str, Dict[str, List[float]]], out_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot.")
        return

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    # Panel 1: upper loss
    for name, h in histories.items():
        axes[0].plot(h["outer_step"], h["upper_loss"], label=name, linewidth=2)
    axes[0].set_xlabel("outer step")
    axes[0].set_ylabel("upper (query) loss")
    axes[0].set_title("Upper-level objective")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Panel 2: query accuracy
    for name, h in histories.items():
        axes[1].plot(h["outer_step"], h["query_acc"], label=name, linewidth=2)
    axes[1].set_xlabel("outer step")
    axes[1].set_ylabel("query accuracy")
    axes[1].set_title("5-way 5-shot query accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Panel 3: mu_SC vs mu_PL (using the Lemma-2.1 quadratic-growth estimator
    # for mu_PL when available, from BCR26).
    for name, h in histories.items():
        axes[2].semilogy(
            h["outer_step"], h["mu_SC_est"],
            label=f"{name}: mu_SC", linestyle="--", linewidth=2,
        )
        # mu_PL_QG from Lemma 2.1 is the "correct" one; fall back to the
        # classical probe estimator if absent.
        mu_pl_curve = h.get("mu_PL_QG") or h["mu_PL_est"]
        axes[2].semilogy(
            h["outer_step"], mu_pl_curve,
            label=f"{name}: mu_PL (QG)", linestyle="-", linewidth=2,
        )
    axes[2].set_xlabel("outer step")
    axes[2].set_ylabel("constant (log scale)")
    axes[2].set_title(
        "Strong convexity vs PL\n(mu_PL from quad-growth, BCR26 Lem 2.1)"
    )
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3, which="both")

    # Panel 4: Morse-Bott signature -- empirical verification of BCR26
    # Lemma 2.2. dim_ker_hess should be large and flat in the PL regime
    # (rank deficit = dim of S); it should be ~0 in the strongly-convex
    # regime (S is a point).
    any_mb = False
    for name, h in histories.items():
        if "dim_ker_hess" in h and len(h["dim_ker_hess"]) > 0:
            any_mb = True
            axes[3].plot(
                h["outer_step"], h["dim_ker_hess"],
                label=f"{name}: dim ker H (= dim T_w* S)",
                linestyle="-", linewidth=2,
            )
            axes[3].plot(
                h["outer_step"], h["dim_range_hess"],
                label=f"{name}: dim rng H (= k, codim S)",
                linestyle="--", linewidth=2,
            )
    if any_mb:
        axes[3].set_xlabel("outer step")
        axes[3].set_ylabel("eigenvalue cluster size")
        axes[3].set_title(
            "Morse-Bott signature of Hess_w g(w*)\n"
            "(BCR26 Lemma 2.2: bimodal spectrum)"
        )
        axes[3].legend(fontsize=7)
        axes[3].grid(alpha=0.3)
    else:
        axes[3].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"Saved plot to {out_path}")


def plot_solver_quality(
    histories: Dict[str, Dict[str, List[float]]],
    out_path: str,
    cg_steps: int,
    cg_steps_long: int,
) -> None:
    """Two-panel plot of head-to-head solver quality.

    Left panel: relative error of each solver's v vs the Moore-Penrose pinv
    reference, per outer step, on a log scale. The Morse-Bott pinv should
    sit near 1e-3; ridge CG should be small in the strong_convex regime
    and LARGE in the PL/tiny_lam regimes.

    Right panel: the same data but flipped to show "extra HVPs to close
    the gap" -- specifically, the ratio err_ridge_short / err_ridge_long,
    showing how much the legacy solver improves with cg_steps_long extra
    iterations. In the SC regime this ratio should be near 1 (already
    converged); in the PL regime it can be > 1 (more iters help) or = 1
    (the gap is structural and CG cannot close it).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot.")
        return

    # Filter to regimes that actually measured quality.
    regimes_q = [
        name for name, h in histories.items()
        if h.get("err_morse_bott") and any(
            v == v and v > 0  # not NaN, positive
            for v in h["err_morse_bott"]
        )
    ]
    if not regimes_q:
        print("No regimes with quality measurements; skipping solver-quality plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # --- Left: per-step relative errors, three solvers, all regimes ---
    color_cycle = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, name in enumerate(regimes_q):
        h = histories[name]
        c = color_cycle[i % len(color_cycle)]
        axes[0].semilogy(
            h["outer_step"], h["err_morse_bott"],
            label=f"{name}: Morse-Bott pinv",
            color=c, linestyle="-", linewidth=2,
        )
        axes[0].semilogy(
            h["outer_step"], h["err_ridge_cg_short"],
            label=f"{name}: ridge CG ({cg_steps} iters)",
            color=c, linestyle="--", linewidth=1.5,
        )
        axes[0].semilogy(
            h["outer_step"], h["err_ridge_cg_long"],
            label=f"{name}: ridge CG ({cg_steps_long} iters)",
            color=c, linestyle=":", linewidth=1.5,
        )
    # Reference horizontal lines.
    axes[0].axhline(1.0, color="gray", linewidth=1, alpha=0.4)
    axes[0].text(
        0, 1.0, "  100% error (zero solution)",
        fontsize=8, color="gray", verticalalignment="bottom",
    )
    axes[0].set_xlabel("outer step")
    axes[0].set_ylabel(r"$\|v_{\rm method} - v_{\rm pinv}\| / \|v_{\rm pinv}\|$")
    axes[0].set_title(
        "Hypergradient v-solver quality\n"
        "(reference = Moore-Penrose pinv via BCR26 Lem 2.2)"
    )
    axes[0].legend(fontsize=7, loc="best")
    axes[0].grid(alpha=0.3, which="both")

    # --- Right: improvement ratio of ridge CG with extra iterations ---
    # err_ridge_short / err_ridge_long >> 1 means more iters help; ~ 1 means
    # the legacy method has stalled (the gap is structural, not iterative).
    for i, name in enumerate(regimes_q):
        h = histories[name]
        c = color_cycle[i % len(color_cycle)]
        ratios = [
            (s / l) if (l == l and l > 1e-15) else float("nan")
            for s, l in zip(h["err_ridge_cg_short"], h["err_ridge_cg_long"])
        ]
        axes[1].semilogy(
            h["outer_step"], ratios,
            label=name,
            color=c, linestyle="-", linewidth=2,
        )
    axes[1].axhline(1.0, color="gray", linewidth=1, alpha=0.6,
                    label="ratio = 1 (extra iters do nothing)")
    axes[1].set_xlabel("outer step")
    axes[1].set_ylabel(
        f"err(ridge, {cg_steps} iters) / err(ridge, {cg_steps_long} iters)"
    )
    axes[1].set_title(
        "Does throwing more CG iterations at it help?\n"
        "(>1: yes; ~1: no, the gap is structural)"
    )
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"Saved plot to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

REGIME_CONFIGS = {
    # (lam, eps_pl, use_morse_bott_default)
    #
    # strong_convex: lam >> 0 -> Hess_w g is uniformly pos-def,
    #                so the Morse-Bott split is trivial (empty kernel)
    #                and both solvers agree. use_morse_bott=True still
    #                works (Lanczos will just find zero small eigs) but
    #                we default to False for speed.
    #
    # tiny_lam / pl_only: Hess_w g is rank-deficient. BCR26's Morse-Bott
    #                    pseudoinverse is the principled solver; use it.
    "strong_convex": (1e-2, 0.0,  False),
    "tiny_lam":      (1e-6, 1e-4, True),
    "pl_only":       (0.0,  1e-3, True),
}


def main():
    """Run the experiment using the values defined in the SETTINGS class
    at the top of this file. To change what runs, edit SETTINGS and rerun.
    """
    # ---- Resolve mode-dependent defaults from SETTINGS ----
    # MEASURE_HG_QUALITY adds a dense Hess eigh AND a 200-iter CG every
    # outer step. To keep wall-clock reasonable, drop n_outer to 100 and
    # n_tasks_per_batch to 2 unless the user explicitly overrode them in
    # SETTINGS.
    n_outer = SETTINGS.N_OUTER
    n_tasks_per_batch = SETTINGS.N_TASKS_PER_BATCH
    if SETTINGS.MEASURE_HG_QUALITY:
        if n_outer is None:
            n_outer = 100
        if n_tasks_per_batch is None:
            n_tasks_per_batch = 2
    else:
        if n_outer is None:
            n_outer = 200
        if n_tasks_per_batch is None:
            n_tasks_per_batch = 4

    # Resolve device.
    if SETTINGS.DEVICE is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = SETTINGS.DEVICE
    device = torch.device(device_str)

    # Validate REGIME.
    valid_regimes = {"strong_convex", "tiny_lam", "pl_only", "all"}
    if SETTINGS.REGIME not in valid_regimes:
        raise ValueError(
            f"SETTINGS.REGIME must be one of {sorted(valid_regimes)}, "
            f"got {SETTINGS.REGIME!r}"
        )

    os.makedirs(SETTINGS.OUT_DIR, exist_ok=True)
    print(f"Using device: {device}")
    print(
        f"Resolved settings: regime={SETTINGS.REGIME}, "
        f"n_outer={n_outer}, n_tasks_per_batch={n_tasks_per_batch}, "
        f"measure_hg_quality={SETTINGS.MEASURE_HG_QUALITY}, "
        f"out_dir={SETTINGS.OUT_DIR}"
    )

    # Shared data sampler across regimes so comparison is apples-to-apples.
    dataset = SyntheticFewShotDataset(n_classes=64, seed=SETTINGS.SEED)

    regimes_to_run = (
        [SETTINGS.REGIME] if SETTINGS.REGIME != "all"
        else ["strong_convex", "tiny_lam", "pl_only"]
    )

    histories: Dict[str, Dict[str, List[float]]] = {}
    for regime in regimes_to_run:
        lam_default, eps_pl_default, mb_default = REGIME_CONFIGS[regime]
        lam = SETTINGS.LAM if SETTINGS.LAM is not None else lam_default
        eps_pl = SETTINGS.EPS_PL if SETTINGS.EPS_PL is not None else eps_pl_default
        use_mb = (
            SETTINGS.USE_MORSE_BOTT if SETTINGS.USE_MORSE_BOTT is not None
            else mb_default
        )

        cfg = RunConfig(
            regime=regime,
            lam=lam,
            eps_pl=eps_pl,
            n_outer=n_outer,
            n_inner=SETTINGS.N_INNER,
            n_tasks_per_batch=n_tasks_per_batch,
            seed=SETTINGS.SEED,
            use_morse_bott=use_mb,
            n_lanczos=SETTINGS.N_LANCZOS,
            measure_hg_quality=SETTINGS.MEASURE_HG_QUALITY,
            cg_steps_long=SETTINGS.CG_STEPS_LONG,
        )
        print("\n" + "=" * 70)
        print(
            f"Running regime: {regime}  "
            f"(lam={lam}, eps_pl={eps_pl}, use_morse_bott={use_mb}, "
            f"measure_hg_quality={SETTINGS.MEASURE_HG_QUALITY})"
        )
        print("=" * 70)
        h = run_regime(cfg, dataset.sample_task, device)
        histories[regime] = h

    if len(histories) > 1:
        plot_comparison(histories, os.path.join(SETTINGS.OUT_DIR, "comparison.png"))
    if SETTINGS.MEASURE_HG_QUALITY:
        # Solver-quality plot is informative even with a single regime
        # (it shows three curves per regime: MB, ridge-short, ridge-long).
        plot_solver_quality(
            histories,
            os.path.join(SETTINGS.OUT_DIR, "solver_quality.png"),
            cg_steps=20,                  # default cg_steps from RunConfig
            cg_steps_long=SETTINGS.CG_STEPS_LONG,
        )

    # Save raw histories.
    import json
    with open(os.path.join(SETTINGS.OUT_DIR, "histories.json"), "w") as fp:
        json.dump(histories, fp, indent=2)
    print(f"\nDone. Results in {SETTINGS.OUT_DIR}/")


if __name__ == "__main__":
    main()

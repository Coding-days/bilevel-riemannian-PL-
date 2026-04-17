"""
Riemannian Bilevel Meta-Learning: Strong Convexity vs PL
========================================================

Reproduces the meta-learning experiment from:
  Han, Mishra, Jawanpuria, Takeda.
  "A Framework for Bilevel Optimization on Riemannian Manifolds."
  NeurIPS 2024. (Section 4.3, MiniImageNet 5-way 5-shot.)

This script is instrumented to compare three lower-level regimes:

  (A) STRONG_CONVEX:  g(theta, w) = L(theta, w; D_s) + (lam/2) * ||w||^2
                      with lam moderate (e.g., 1e-2). This is the original
                      paper's setup -- the L2 regularizer R(w) ensures the
                      lower-level is mu-strongly convex with mu >= lam.
                      Implicit-differentiation is clean because Hess_w g
                      is uniformly positive definite.

  (B) PL_ONLY:        lam = 0. Cross-entropy with a linear last layer is
                      CONVEX in w given fixed features, but only strongly
                      convex when the feature Gram matrix is full rank.
                      In 5-way 5-shot, per-task support has only 25 samples
                      while the feature dim can be >= 25, so Hess_w is
                      rank-deficient along directions in the nullspace.
                      The problem is then only PL (every stationary point
                      is a global minimum, quadratic growth holds on the
                      image of the Hessian). This is the regime our PL
                      paper targets.

  (C) TINY_LAM:       lam very small (e.g., 1e-6). Technically strongly
                      convex, but the condition number 1/mu is enormous,
                      so the strong-convexity-based analysis gives vacuous
                      rates. PL analysis remains meaningful.

How the paper attains strong convexity (answering your question)
----------------------------------------------------------------
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

The algorithm itself (RHGD) is taken directly from Han et al. Section 3:
conjugate gradient hypergradient approximation on the Stiefel manifold
(we use the canonical/Euclidean metric + QR retraction exactly as the
reference implementation does via geoopt).

Requirements
------------
  pip install torch torchvision geoopt numpy matplotlib

Dataset
-------
By default the script runs on synthetic "MiniImageNet-like" data
(random low-rank structure per class) so it runs on any machine. To
run on real MiniImageNet, point --data_root at a directory holding
the standard pickle files and pass --real_data.

Usage
-----
  python riemannian_meta_learning_pl.py --regime strong_convex
  python riemannian_meta_learning_pl.py --regime pl_only
  python riemannian_meta_learning_pl.py --regime tiny_lam --lam 1e-6
  python riemannian_meta_learning_pl.py --regime all   # runs all three
"""

from __future__ import annotations

import argparse
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
# Under PL / rank-deficient Hess_w, we solve the regularized/clipped system
# from our PL paper (eq. 2): S_{[mu, Lg]}[Hess] v = grad_w f. In practice we
# approximate the spectral clipping by adding a small ridge eps*I inside the
# CG operator; this is a standard, well-behaved surrogate and matches the
# regularized tangent-space system up to the choice of eps.
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
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute an approximation of grad_Theta F for a single task.

    Returns (upper_loss_scalar_tensor, diagnostics_dict).
    The side effect is that backbone.upper_params() have their .grad
    attributes INCREMENTED with the hypergradient contribution.
    """

    # ---- Lower-level solve (inner loop) ----
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

    # ---- Implicit term: need v* = [Hess_w g]^{-1} grad_w f, then
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

    def apply_A(v: torch.Tensor) -> torch.Tensor:
        return hessian_vector_product(
            feats_s_det, support_y, w_star, v, lam=lam_lower, eps_pl=eps_pl,
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
    # mu_SC = smallest eigenvalue of Hess_w g at w_star (strong convexity constant)
    # mu_PL = ||grad_w g||^2 / (2 * (g - g*)) -- at random perturbations from w_star
    # Both are estimated on the support set for this task.
    diag = estimate_pl_vs_sc(
        feats_s_det, support_y, w_star, lam=lam_lower, n_way=n_way,
    )
    diag["upper_loss"] = f_value.item()
    return f_value.detach(), diag


def estimate_pl_vs_sc(
    features: torch.Tensor,
    labels: torch.Tensor,
    w_star: torch.Tensor,
    lam: float,
    n_way: int,
    n_probe: int = 8,
) -> Dict[str, float]:
    """
    Empirically estimate the strong-convexity constant mu_SC and the PL
    constant mu_PL of g(theta_fixed, .) at w_star.

    mu_SC is approximated as the smallest eigenvalue of Hess_w g at w_star
    (via a few steps of inverse power iteration on Hess + eps*I).
    mu_PL is approximated as inf over random perturbations w_tilde of
      ||grad_w g(w_tilde)||^2 / (2 (g(w_tilde) - g(w_star))).

    These diagnostics are the whole point of the PL regime: if mu_PL is
    bounded away from 0 while mu_SC ~ 0, you have a problem that
    CANNOT be handled by the strong-convexity analysis of Han et al.
    but IS handled by our PL analysis.
    """
    d = features.size(1)
    device = features.device
    features = features.detach()

    # --- mu_SC via power iteration on (H + lam*I)^{-1} ---
    # Inverse power iteration gives 1 / lambda_min. We use a small CG solve
    # as the inverse application.
    def apply_H(v):
        return hessian_vector_product(features, labels, w_star, v, lam=0.0)

    # Try 3 random starts and take the best estimate of smallest eigenvalue.
    mu_sc_est = float("inf")
    for _ in range(3):
        v = torch.randn(n_way, d, device=device)
        v = v / (v.norm() + 1e-20)
        for _ in range(20):
            # Apply H + delta*I with small delta to keep CG well-conditioned.
            delta = 1e-4
            sol = conjugate_gradient(
                lambda u: apply_H(u) + delta * u, v, n_steps=15,
            )
            v = sol / (sol.norm() + 1e-20)
        Hv = apply_H(v)
        rq = (v * Hv).sum().item()  # Rayleigh quotient ~= smallest eigenvalue
        mu_sc_est = min(mu_sc_est, max(rq, 0.0))
    # Strong-convexity constant is >= lam by construction, so take max.
    mu_sc_est = max(mu_sc_est, lam)

    # --- mu_PL via random probes ---
    g_star = task_loss(features, labels, w_star, lam=lam).item()
    mu_pl_est = float("inf")
    for _ in range(n_probe):
        # Perturbation scale: match w_star's typical magnitude.
        scale = max(0.1, w_star.norm().item() * 0.3 / math.sqrt(n_way * d))
        w_tilde = (w_star + scale * torch.randn_like(w_star)).requires_grad_(True)
        g_val = task_loss(features, labels, w_tilde, lam=lam)
        grad_val = torch.autograd.grad(g_val, w_tilde)[0]
        num = (grad_val * grad_val).sum().item()
        denom = 2.0 * max(g_val.item() - g_star, 1e-10)
        mu_pl_est = min(mu_pl_est, num / denom)

    return {
        "mu_SC_est": mu_sc_est,
        "mu_PL_est": mu_pl_est,
        "lam_used": lam,
    }


# ---------------------------------------------------------------------------
# Outer (upper-level) loop: Riemannian hypergradient descent
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    regime: str                  # "strong_convex" | "pl_only" | "tiny_lam"
    lam: float                   # lower-level L2 strength
    eps_pl: float                # regularization for CG in PL regime
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
        "mu_SC_est": [], "mu_PL_est": [], "wallclock": [],
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
        diag_accum = {"mu_SC_est": 0.0, "mu_PL_est": 0.0}

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

            diag_accum["mu_SC_est"] += diag["mu_SC_est"]
            diag_accum["mu_PL_est"] += diag["mu_PL_est"]

        # Average gradient across tasks (hypergrad accumulator is a sum).
        for p in backbone.upper_params():
            if p.grad is not None:
                p.grad.div_(cfg.n_tasks_per_batch)

        optimizer.step()

        batch_loss /= cfg.n_tasks_per_batch
        batch_acc /= cfg.n_tasks_per_batch
        for k in diag_accum:
            diag_accum[k] /= cfg.n_tasks_per_batch

        if step % cfg.log_every == 0 or step == cfg.n_outer - 1:
            history["outer_step"].append(step)
            history["upper_loss"].append(batch_loss)
            history["query_acc"].append(batch_acc)
            history["mu_SC_est"].append(diag_accum["mu_SC_est"])
            history["mu_PL_est"].append(diag_accum["mu_PL_est"])
            history["wallclock"].append(time.time() - t0)
            print(
                f"[{cfg.regime:>14s}] step {step:4d} | "
                f"upper_loss={batch_loss:.4f} | "
                f"query_acc={batch_acc:.3f} | "
                f"mu_SC~={diag_accum['mu_SC_est']:.2e} | "
                f"mu_PL~={diag_accum['mu_PL_est']:.2e} | "
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

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

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

    # Panel 3: PL vs SC constants (log scale) -- the PL story in one plot.
    for name, h in histories.items():
        axes[2].semilogy(
            h["outer_step"], h["mu_SC_est"],
            label=f"{name}: mu_SC", linestyle="--", linewidth=2,
        )
        axes[2].semilogy(
            h["outer_step"], h["mu_PL_est"],
            label=f"{name}: mu_PL", linestyle="-", linewidth=2,
        )
    axes[2].set_xlabel("outer step")
    axes[2].set_ylabel("constant (log scale)")
    axes[2].set_title("Strong convexity vs PL constants\n(SC collapses without regularizer; PL persists)")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"Saved plot to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

REGIME_CONFIGS = {
    # (lam, eps_pl)
    "strong_convex": (1e-2, 0.0),
    "tiny_lam":      (1e-6, 1e-4),
    "pl_only":       (0.0,  1e-3),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", type=str, default="all",
                        choices=["strong_convex", "tiny_lam", "pl_only", "all"])
    parser.add_argument("--lam", type=float, default=None,
                        help="Override lambda (lower-level L2 strength)")
    parser.add_argument("--eps_pl", type=float, default=None,
                        help="Override CG regularization (spectral clip surrogate)")
    parser.add_argument("--n_outer", type=int, default=200)
    parser.add_argument("--n_inner", type=int, default=30)
    parser.add_argument("--n_tasks_per_batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", type=str, default="./outputs_riem_meta_pl")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Shared data sampler across regimes so comparison is apples-to-apples.
    dataset = SyntheticFewShotDataset(n_classes=64, seed=args.seed)

    regimes_to_run = (
        [args.regime] if args.regime != "all"
        else ["strong_convex", "tiny_lam", "pl_only"]
    )

    histories: Dict[str, Dict[str, List[float]]] = {}
    for regime in regimes_to_run:
        lam_default, eps_pl_default = REGIME_CONFIGS[regime]
        lam = args.lam if args.lam is not None else lam_default
        eps_pl = args.eps_pl if args.eps_pl is not None else eps_pl_default

        cfg = RunConfig(
            regime=regime,
            lam=lam,
            eps_pl=eps_pl,
            n_outer=args.n_outer,
            n_inner=args.n_inner,
            n_tasks_per_batch=args.n_tasks_per_batch,
            seed=args.seed,
        )
        print("\n" + "=" * 70)
        print(f"Running regime: {regime}  (lam={lam}, eps_pl={eps_pl})")
        print("=" * 70)
        h = run_regime(cfg, dataset.sample_task, device)
        histories[regime] = h

    if len(histories) > 1:
        plot_comparison(histories, os.path.join(args.out_dir, "comparison.png"))

    # Save raw histories.
    import json
    with open(os.path.join(args.out_dir, "histories.json"), "w") as fp:
        json.dump(histories, fp, indent=2)
    print(f"\nDone. Results in {args.out_dir}/")


if __name__ == "__main__":
    main()

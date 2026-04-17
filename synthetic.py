"""
Comparison: RHGD (Han et al. 2024) vs R-HJFBiO (Ours)
=======================================================
Bilevel optimization on Stiefel manifold x Euclidean space.

Upper: min_{W in St(d,r)} f(W, y*(W))
Lower: y*(W) = argmin_y g(W, y)

Two settings:
  (A) g is strongly convex  =>  both methods work
  (B) g satisfies PL but NOT strongly convex (overparameterized)
      =>  Han et al. breaks (Hessian singular), ours works
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import geoopt
import time

torch.manual_seed(42)
np.random.seed(42)


class BilevelProblem:
    """Bilevel problem on St(d,r) x R^p."""

    def __init__(self, d, r, p, setting="sc"):
        self.d, self.r, self.p = d, r, p
        self.setting = setting
        self.stiefel = geoopt.Stiefel()

        # Fixed data
        X0 = torch.randn(d, d) * 0.1
        self.X = X0 @ X0.T + 0.01 * torch.eye(d)
        self.Y = torch.randn(d, r) * 0.5
        self.c = torch.randn(p)

        if setting == "sc":
            # g(W,y) = 0.5*||Ay - b(W)||^2 + (mu/2)*||y||^2
            # H_y g = A^T A + mu I  (positive definite, invertible)
            self.mu_reg = 0.1
            A0 = torch.randn(p, p)
            U, _, Vt = torch.linalg.svd(A0, full_matrices=False)
            S = torch.linspace(1.0, 3.0, p)
            self.A = U @ torch.diag(S) @ Vt
        else:
            # g(W,y) = 0.5*||Ay - b(W)||^2,  A fat (m < p)
            # H_y g = A^T A  has nullspace => NOT strongly convex
            # PL holds with mu = sigma_min^+(A)^2
            self.mu_reg = 0.0
            m = max(p // 3, 2)
            A0 = torch.randn(m, p)
            U, _, Vt = torch.linalg.svd(A0, full_matrices=False)
            S = torch.linspace(0.5, 2.0, m)
            self.A = U @ torch.diag(S) @ Vt
            self.pl_mu = S.min().item() ** 2

    def b(self, W):
        WtXW = W.T @ self.X @ W
        bvec = WtXW.flatten()
        m = self.A.shape[0]
        if bvec.shape[0] >= m:
            return bvec[:m]
        return torch.nn.functional.pad(bvec, (0, m - bvec.shape[0]))

    def g(self, W, y):
        res = self.A @ y - self.b(W)
        val = 0.5 * res.dot(res)
        if self.mu_reg > 0:
            val = val + 0.5 * self.mu_reg * y.dot(y)
        return val

    def f(self, W, y):
        return (y - self.c).dot(y - self.c) + 0.1 * torch.trace(W.T @ self.X @ self.Y)

    def grad_y_g(self, W, y):
        y_ = y.clone().requires_grad_(True)
        self.g(W, y_).backward()
        return y_.grad.detach()

    def grad_y_f(self, W, y):
        y_ = y.clone().requires_grad_(True)
        self.f(W, y_).backward()
        return y_.grad.detach()

    def grad_x_f(self, W, y):
        W_ = W.clone().requires_grad_(True)
        self.f(W_, y).backward()
        return W_.grad.detach()

    def hess_y_g(self, W, y):
        H = self.A.T @ self.A
        if self.mu_reg > 0:
            H = H + self.mu_reg * torch.eye(self.p)
        return H

    def hypergradient_hinv(self, W, y):
        """Han et al. HINV: grad_x f - J^T H^{-1} grad_y f"""
        H = self.hess_y_g(W, y)
        gyf = self.grad_y_f(W, y)

        # Check conditioning
        eigvals = torch.linalg.eigvalsh(H)
        if eigvals.min() < 1e-8:
            return None  # singular

        v = torch.linalg.solve(H, gyf)
        if torch.any(torch.isnan(v)):
            return None

        # Cross-deriv via autograd
        W_ = W.clone().requires_grad_(True)
        y_ = y.clone().requires_grad_(True)
        gval = self.g(W_, y_)
        gy = torch.autograd.grad(gval, y_, create_graph=True)[0]
        Jv = torch.autograd.grad(gy, W_, grad_outputs=v)[0]

        return self.grad_x_f(W, y) - Jv

    def hypergradient_hjfbio(self, W, y, v_aux, mu_clip=0.1, delta_eps=1e-4):
        """Our R-HJFBiO: spectral clipping + finite differences."""
        H = self.hess_y_g(W, y)
        eigvals, eigvecs = torch.linalg.eigh(H)
        clipped = torch.clamp(eigvals, min=mu_clip, max=10.0)
        H_clip = eigvecs @ torch.diag(clipped) @ eigvecs.T

        gyf = self.grad_y_f(W, y)

        # Jacobian-free cross term via finite diff
        W1 = W.clone().requires_grad_(True)
        g1 = self.g(W1, y + delta_eps * v_aux)
        J1 = torch.autograd.grad(g1, W1)[0]

        W2 = W.clone().requires_grad_(True)
        g2 = self.g(W2, y - delta_eps * v_aux)
        J2 = torch.autograd.grad(g2, W2)[0]

        Jv = (J1 - J2) / (2 * delta_eps)
        w = self.grad_x_f(W, y) - Jv

        # Auxiliary update direction
        h = H_clip @ v_aux - gyf
        return w, h


def rhgd_han(prob, K=300, S=15, eta_x=0.005, eta_y=0.03):
    """RHGD-HINV (Han et al. 2024)."""
    mf = prob.stiefel
    W, _ = torch.linalg.qr(torch.randn(prob.d, prob.r))
    y = torch.randn(prob.p) * 0.1
    hist = {"F": [], "grad_norm": [], "time": []}
    t0 = time.time()

    for k in range(K):
        # Inner loop: lower-level GD
        yk = y.clone()
        for _ in range(S):
            yk = yk - eta_y * prob.grad_y_g(W, yk)
        y = yk

        hyper = prob.hypergradient_hinv(W, y)
        if hyper is None:
            print(f"  RHGD-HINV: Hessian singular at iter {k}!")
            for _ in range(k, K):
                hist["F"].append(float('nan'))
                hist["grad_norm"].append(float('nan'))
                hist["time"].append(time.time() - t0)
            return hist

        rg = mf.egrad2rgrad(W, hyper)
        hist["F"].append(prob.f(W, y).item())
        hist["grad_norm"].append(torch.norm(rg).item())
        hist["time"].append(time.time() - t0)

        W = mf.retr(W, -eta_x * rg)

    return hist


def rhjfbio_ours(prob, T=300, S=15, gamma=0.005, lam=0.03, tau=0.01,
                 mu_clip=0.1):
    """R-HJFBiO (Ours)."""
    mf = prob.stiefel
    W, _ = torch.linalg.qr(torch.randn(prob.d, prob.r))
    y = torch.randn(prob.p) * 0.1
    v = torch.zeros(prob.p)
    hist = {"F": [], "grad_norm": [], "time": []}
    t0 = time.time()

    for t in range(T):
        y_new = y - lam * prob.grad_y_g(W, y)
        w, h = prob.hypergradient_hjfbio(W, y_new, v, mu_clip=mu_clip)

        rg = mf.egrad2rgrad(W, w)
        hist["F"].append(prob.f(W, y_new).item())
        hist["grad_norm"].append(torch.norm(rg).item())
        hist["time"].append(time.time() - t0)

        W = mf.retr(W, -gamma * rg)
        v = v - tau * h
        rv = 5.0
        if torch.norm(v) > rv:
            v = v * (rv / torch.norm(v))
        y = y_new

    return hist


# ============================================================
# Run
# ============================================================

print("Running experiments...\n")

# (A) Strongly convex
print("=" * 55)
print("Setting A: g is strongly convex")
print("=" * 55)
prob_sc = BilevelProblem(10, 3, 12, "sc")
H = prob_sc.hess_y_g(torch.eye(10, 3), torch.zeros(12))
eigs = torch.linalg.eigvalsh(H)
print(f"  H_y g eigenvalues: [{eigs.min():.3f}, {eigs.max():.3f}], kappa={eigs.max()/eigs.min():.1f}")

torch.manual_seed(42); h_sc_han = rhgd_han(prob_sc)
torch.manual_seed(42); h_sc_ours = rhjfbio_ours(prob_sc)

# (B) PL only
print("\n" + "=" * 55)
print("Setting B: g satisfies PL but is NOT strongly convex")
print("=" * 55)
prob_pl = BilevelProblem(10, 3, 12, "pl")
H = prob_pl.hess_y_g(torch.eye(10, 3), torch.zeros(12))
eigs = torch.linalg.eigvalsh(H)
print(f"  H_y g eigenvalues (first 6): {[f'{e:.4f}' for e in eigs[:6].tolist()]}")
print(f"  Zero eigenvalues: {(eigs < 1e-8).sum().item()}, PL mu={prob_pl.pl_mu:.4f}")

torch.manual_seed(42); h_pl_han = rhgd_han(prob_pl)
torch.manual_seed(42); h_pl_ours = rhjfbio_ours(prob_pl)


# ============================================================
# Plot
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
colors = {"han": "#d62728", "ours": "#1f77b4"}

def plot_series(ax, han, ours, ylabel, title):
    has_nan = any(np.isnan(v) for v in han)
    if has_nan:
        valid = [(i, v) for i, v in enumerate(han) if not np.isnan(v)]
        if valid:
            ax.semilogy([i for i, _ in valid], [v for _, v in valid],
                        label="RHGD-HINV (Han et al.) — FAILED",
                        color=colors["han"], lw=2, ls="--")
            ax.axvline(x=valid[-1][0], color=colors["han"], ls=":", alpha=0.6)
            ax.annotate("Hessian singular!", xy=(valid[-1][0], valid[-1][1]),
                        xytext=(valid[-1][0]+15, valid[-1][1]),
                        fontsize=9, color=colors["han"], fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color=colors["han"]))
    else:
        ax.semilogy(han, label="RHGD-HINV (Han et al.)", color=colors["han"], lw=2)

    ax.semilogy(ours, label="R-HJFBiO (Ours)", color=colors["ours"], lw=2)
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

plot_series(axes[0,0], h_sc_han["F"], h_sc_ours["F"],
            "F(W, y)", "(A) Strongly Convex g — Objective")
plot_series(axes[0,1], h_sc_han["grad_norm"], h_sc_ours["grad_norm"],
            "‖∇F‖", "(A) Strongly Convex g — Gradient Norm")
plot_series(axes[1,0], h_pl_han["F"], h_pl_ours["F"],
            "F(W, y)", "(B) PL (Not S.C.) g — Objective")
plot_series(axes[1,1], h_pl_han["grad_norm"], h_pl_ours["grad_norm"],
            "‖∇F‖", "(B) PL (Not S.C.) g — Gradient Norm")

fig.suptitle("Bilevel Optimization on St(d,r): RHGD-HINV vs R-HJFBiO",
             fontsize=14, fontweight="bold", y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("outputs/comparison_plot.png", dpi=150, bbox_inches="tight")
plt.close()

# Summary
print("\n" + "=" * 55)
print("SUMMARY")
print("=" * 55)
print(f"\n(A) Strongly Convex:")
print(f"    Han  final F={h_sc_han['F'][-1]:.4f}  |∇F|={h_sc_han['grad_norm'][-1]:.4f}")
print(f"    Ours final F={h_sc_ours['F'][-1]:.4f}  |∇F|={h_sc_ours['grad_norm'][-1]:.4f}")
nan_han = any(np.isnan(v) for v in h_pl_han["F"])
print(f"\n(B) PL (not s.c.):")
if nan_han:
    n_valid = sum(1 for v in h_pl_han["F"] if not np.isnan(v))
    print(f"    Han  FAILED at iter {n_valid} (singular Hessian)")
else:
    print(f"    Han  final F={h_pl_han['F'][-1]:.4f}  |∇F|={h_pl_han['grad_norm'][-1]:.4f}")
print(f"    Ours final F={h_pl_ours['F'][-1]:.4f}  |∇F|={h_pl_ours['grad_norm'][-1]:.4f}")
print("\nDone!")

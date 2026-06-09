"""
sinkhorn.py
-----------
Entropic relaxation of the martingale optimal transport problem, solved via
a Sinkhorn-type algorithm (Henry-Labordère, 2019, §2.2).

Primal problem (regularised MOT):
    MK_ε(µ1, µ2) = sup_{P ∈ M(µ1, µ2)}  E^P[c] − ε H(P | ρ0)

Dual problem:
    inf_{u1, u2, h}  E^µ1[u1] + E^µ2[u2]
        + ε E^ρ0[ exp((c − u1 − u2 − h·(S2−S1)) / ε) ]

The algorithm alternates updates of u1, h, and u2 using the first-order
optimality conditions.  Log-sum-exp is used throughout for numerical
stability.

References
----------
[2]  Henry-Labordère (2019).  arXiv:1904.04546.
[10] Cuturi (2013) — Sinkhorn distances.
[12] Guo & Obłój (2019) — Sinkhorn for MOT.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Discretisation helper
# ---------------------------------------------------------------------------

def discretize(dist, a: float, b: float, n: int):
    """
    Discretize a scipy.stats distribution on [a, b] using n atoms.

    Atoms are placed at midpoints of n equal-width sub-intervals.
    Weights are computed from CDF differences (mass-preserving).

    Parameters
    ----------
    dist : scipy.stats frozen distribution
    a, b : support bounds
    n    : number of atoms

    Returns
    -------
    atoms   : (n,) atom positions
    weights : (n,) probability masses (sum to 1)
    """
    edges   = np.linspace(a, b, n + 1)
    atoms   = 0.5 * (edges[:-1] + edges[1:])
    weights = dist.cdf(edges[1:]) - dist.cdf(edges[:-1])
    total   = weights.sum()
    weights = weights / total if total > 0 else np.ones(n) / n
    return atoms, weights


# ---------------------------------------------------------------------------
# Dual potential updates
# ---------------------------------------------------------------------------

def _update_u1(G, D, u2, h, beta, eps):
    """Update u1 from optimality condition (Eq. 5 in HL 2019)."""
    A   = (G - u2[None, :] - h[:, None] * D) / eps   # (m1, m2)
    u1  = eps * logsumexp(np.log(beta)[None, :] + A, axis=1)
    return u1


def _update_h(G, D, u2, beta, eps, x):
    """
    Update h row-by-row via root-finding (Eq. 7 in HL 2019).

    For each s1_i, find θ such that
        Σ_j  β_j · (y_j − x_i) · exp((G_ij − u2_j − θ·D_ij) / ε) = 0
    """
    m1      = len(x)
    h_new   = np.zeros(m1)
    log_b   = np.log(beta)
    bracket = (-80.0, 80.0)

    for i in range(m1):
        g_row  = G[i]
        d_row  = D[i]
        base   = (g_row - u2) / eps

        def f(theta):
            return (beta * d_row * np.exp(base - theta * d_row / eps)).sum()

        fa, fb = f(bracket[0]), f(bracket[1])
        if fa * fb > 0:
            h_new[i] = 0.0
        else:
            h_new[i] = brentq(f, bracket[0], bracket[1], xtol=1e-10, maxiter=300)

    return h_new


def _update_u2(G, D, u1, h, alpha, eps):
    """Update u2 from optimality condition (Eq. 6 in HL 2019)."""
    B  = (G - u1[:, None] - h[:, None] * D) / eps   # (m1, m2)
    u2 = eps * logsumexp(np.log(alpha)[:, None] + B, axis=0)
    return u2


# ---------------------------------------------------------------------------
# Optimal plan reconstruction
# ---------------------------------------------------------------------------

def _reconstruct_plan(G, D, u1, u2, h, alpha, beta, eps):
    """
    p*[i,j] = α_i β_j exp((G_ij − u1_i − u2_j − h_i D_ij) / ε)
    Computed in log-space to avoid overflow.
    """
    log_p = (
        np.log(alpha[:, None]) + np.log(beta[None, :])
        + (G - u1[:, None] - u2[None, :] - h[:, None] * D) / eps
    )
    return np.exp(log_p)


def _constraint_errors(P, alpha, beta, x, y):
    """Max absolute errors for µ1 marginal, µ2 marginal, and martingale."""
    e1 = np.max(np.abs(P.sum(axis=1) - alpha))
    e2 = np.max(np.abs(P.sum(axis=0) - beta))
    e3 = np.max(np.abs((P * y[None, :]).sum(axis=1) - alpha * x))
    return e1, e2, e3


# ---------------------------------------------------------------------------
# Main Sinkhorn loop
# ---------------------------------------------------------------------------

def solve(
    x_atoms: np.ndarray,
    alpha: np.ndarray,
    y_atoms: np.ndarray,
    beta: np.ndarray,
    G_mat: np.ndarray,
    *,
    eps: float = 0.005,
    n_iter: int = 1000,
    tol: float = 1e-7,
    verbose: bool = True,
) -> tuple[np.ndarray, float, dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sinkhorn algorithm for entropic MOT (HL 2019 §2.2).

    Parameters
    ----------
    x_atoms, alpha : (m1,) atoms and weights for µ1
    y_atoms, beta  : (m2,) atoms and weights for µ2
    G_mat          : (m1, m2) payoff matrix
    eps            : entropic regularisation parameter
    n_iter         : maximum iterations
    tol            : convergence tolerance on max constraint error
    verbose        : print convergence info

    Returns
    -------
    P       : (m1, m2) optimal transport plan
    value   : float  primal objective E^P[c]
    history : dict with keys 'err_mu1', 'err_mu2', 'err_mart', 'value'
    u1, u2, h : optimal dual potentials
    """
    D = y_atoms[None, :] - x_atoms[:, None]   # (m1, m2) displacement matrix

    u1 = np.zeros(len(x_atoms))
    u2 = np.zeros(len(y_atoms))
    h  = np.zeros(len(x_atoms))

    history = {"err_mu1": [], "err_mu2": [], "err_mart": [], "value": []}

    for it in range(n_iter):
        u1 = _update_u1(G_mat, D, u2, h, beta,   eps)
        h  = _update_h (G_mat, D, u2,    beta,   eps, x_atoms)
        u2 = _update_u2(G_mat, D, u1, h, alpha,  eps)

        P = _reconstruct_plan(G_mat, D, u1, u2, h, alpha, beta, eps)
        e1, e2, e3 = _constraint_errors(P, alpha, beta, x_atoms, y_atoms)
        val = float((P * G_mat).sum())

        history["err_mu1"].append(e1)
        history["err_mu2"].append(e2)
        history["err_mart"].append(e3)
        history["value"].append(val)

        if max(e1, e2, e3) < tol:
            if verbose:
                print(f"  Converged at iteration {it + 1}")
                print(f"  µ1 marginal error : {e1:.2e}")
                print(f"  µ2 marginal error : {e2:.2e}")
                print(f"  Martingale error  : {e3:.2e}")
            break
    else:
        if verbose:
            print(f"  Did not converge in {n_iter} iterations")
            print(f"  Final — µ1: {e1:.2e}, µ2: {e2:.2e}, mart: {e3:.2e}")

    return P, val, history, u1, u2, h


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_results(
    x_atoms: np.ndarray,
    y_atoms: np.ndarray,
    P: np.ndarray,
    history: dict,
    eps: float,
    *,
    title: str = "Sinkhorn MOT",
    save_path: str | None = None,
):
    """
    Three-panel figure: constraint convergence, primal value, transport plan.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(f"{title}  (ε = {eps})", fontsize=13)

    # (A) Constraint errors
    ax = axes[0]
    ax.semilogy(history["err_mu1"],  lw=2, color="steelblue",  label="µ1 marginal")
    ax.semilogy(history["err_mu2"],  lw=2, color="darkorange", label="µ2 marginal")
    ax.semilogy(history["err_mart"], lw=2, color="crimson",    label="Martingale")
    ax.axhline(1e-7, color="grey", ls="--", lw=1, label="tol = 1e-7")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Max constraint error")
    ax.set_title("Constraint convergence")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (B) Primal value
    ax = axes[1]
    ax.plot(history["value"], lw=2, color="steelblue", label="$E^P[c]$")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("$E^P[c(S_1, S_2)]$")
    ax.set_title("Primal value")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (C) Transport plan
    ax = axes[2]
    im = ax.imshow(
        P.T, origin="lower", aspect="auto",
        extent=[x_atoms[0], x_atoms[-1], y_atoms[0], y_atoms[-1]],
        cmap="Blues", interpolation="nearest", alpha=0.85,
    )
    ax.set_xlabel("$s_1$")
    ax.set_ylabel("$s_2$")
    ax.set_title("Optimal transport plan")
    plt.colorbar(im, ax=ax)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Figure saved → {save_path}")

    plt.show()

"""
semi_dual.py
------------
MOT semi-dual solver via neural network parametrisation.

Theory
------
The Kantorovich dual of the MOT problem (Beiglböck et al., 2013) gives:

    MK_c = inf_{u, h}  E^µ2[u(S2)] + E^µ1[φ(S1)]

    φ(s1) = sup_{s2} { c(s1, s2) − u(s2) − h(s1)·(s2 − s1) }

This is a convex minimisation [Boyd & Vandenberghe, 2004].
Gradients are computed via Danskin's theorem [Danskin, 1967]:
    ∇_θ sup_y g(θ, y) = ∇_θ g(θ, y*(θ))

Both u and h are parametrised by independent MLPs (see neural_net.py).

References
----------
[1] Beiglböck, Henry-Labordère, Penkner (2013).
[2] Henry-Labordère (2019).  arXiv:1904.04546.
[5] Danskin (1967).
[6] Kingma & Ba (2015) — Adam.
[13] Boyd & Vandenberghe (2004) — convexity of sup.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .neural_net import MLP


# ---------------------------------------------------------------------------
# c-martingale transform
# ---------------------------------------------------------------------------

def mart_c_transform(
    u_net: MLP,
    h_net: MLP,
    cost_fn,
    s1_batch: np.ndarray,
    s2_pool: np.ndarray,
):
    """
    Compute the martingale c-transform [1, 4]:

        φ(s1_i) = max_j { c(s1_i, s2_j) − u(s2_j) − h(s1_i)·(s2_j − s1_i) }

    The supremum is approximated by a maximum over K pool points [2].

    Parameters
    ----------
    u_net    : MLP parametrising u
    h_net    : MLP parametrising h
    cost_fn  : callable c(s1, s2) → array
    s1_batch : (N, 1) samples from µ1
    s2_pool  : (K, 1) samples from µ2 used to discretise the sup

    Returns
    -------
    phi    : (N, 1)  φ(s1_i) values
    s2star : (N, 1)  argmax s2* for each s1_i
    """
    N = s1_batch.shape[0]
    u_vals = u_net.forward(s2_pool)                        # (K, 1)
    h_vals = h_net.forward(s1_batch)                       # (N, 1)
    C      = cost_fn(s1_batch, s2_pool.T)                  # (N, K)
    penalty = h_vals * (s2_pool.T - s1_batch)              # (N, K)
    obj    = C - u_vals.T - penalty                        # (N, K)

    best_j = np.argmax(obj, axis=1)                        # (N,)
    phi    = obj[np.arange(N), best_j].reshape(-1, 1)      # (N, 1)
    s2star = s2_pool[best_j]                               # (N, 1)
    return phi, s2star


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def solve(
    sample_mu1,
    sample_mu2,
    cost_fn,
    *,
    n_iter: int = 5000,
    batch_size: int = 128,
    pool_size: int = 512,
    lr_u: float = 5e-4,
    lr_h: float = 5e-4,
    hidden_size: int = 32,
    eval_every: int = 500,
    n_eval: int = 4096,
    best_window: int = 1000,
    seed: int = 42,
) -> tuple[MLP, MLP, dict]:
    """
    Solve the MOT semi-dual problem by minimising L(u, h) with Adam.

    Parameters
    ----------
    sample_mu1  : callable n → (n, 1) array   samples from µ1
    sample_mu2  : callable n → (n, 1) array   samples from µ2
    cost_fn     : callable c(s1, s2) → array
    n_iter      : total training iterations
    batch_size  : mini-batch size N
    pool_size   : pool size K for discretising the c-transform sup
    lr_u, lr_h  : Adam learning rates for u and h networks
    hidden_size : width of each hidden layer
    eval_every  : log every this many iterations
    n_eval      : size of fixed evaluation set
    best_window : number of final iterations for best-model selection
    seed        : random seed

    Returns
    -------
    u_net   : trained MLP for u
    h_net   : trained MLP for h
    history : dict with keys 'it', 'L', 'mart_err'
    """
    np.random.seed(seed)
    u_net = MLP(hidden_size)
    h_net = MLP(hidden_size)

    # Fixed evaluation set
    np.random.seed(seed - 1)
    s1_eval = sample_mu1(n_eval)
    s2_eval = sample_mu2(n_eval)
    np.random.seed(seed)

    history = {"it": [], "L": [], "mart_err": []}

    best_mart = float("inf")
    best_u_w: dict | None = None
    best_h_w: dict | None = None

    for it in range(1, n_iter + 1):
        s1       = sample_mu1(batch_size)
        s2       = sample_mu2(batch_size)
        s2_pool  = sample_mu2(pool_size)

        phi, s2star = mart_c_transform(u_net, h_net, cost_fn, s1, s2_pool)

        # --- Gradient w.r.t. θ_u via Danskin [5] ---
        # ∂L/∂θ_u = E_µ2[∂u/∂θ] − E_µ1[∂u/∂θ evaluated at s2*(s1)]
        u_net.forward(s2)
        u_net.backward(np.ones((batch_size, 1)))
        grad_u_pos = {p: u_net.grads[p].copy() for p in u_net._param_names}

        u_net.forward(s2star)
        u_net.backward(-np.ones((batch_size, 1)))

        for p in u_net._param_names:
            u_net.grads[p] = grad_u_pos[p] + u_net.grads[p]
        u_net.step(lr_u, sign=1.0)

        # --- Gradient w.r.t. θ_h via Danskin [5] ---
        # ∂L/∂θ_h = −E_µ1[(s2*(s1) − s1) · ∂h/∂θ]
        residual = s2star - s1
        h_net.forward(s1)
        h_net.backward(-residual)
        h_net.step(lr_h, sign=1.0)

        # --- Best-model tracking (last `best_window` iterations) ---
        if it > n_iter - best_window:
            _, s2star_ev = mart_c_transform(u_net, h_net, cost_fn, s1_eval, s2_eval)
            mart_err = float(np.abs(s2star_ev - s1_eval).mean())
            if mart_err < best_mart:
                best_mart = mart_err
                best_u_w  = u_net.get_weights()
                best_h_w  = h_net.get_weights()

        # --- Periodic evaluation ---
        if it % eval_every == 0 or it == 1:
            phi_ev, s2star_ev = mart_c_transform(u_net, h_net, cost_fn, s1_eval, s2_eval)
            u_ev  = u_net.forward(s2_eval)
            L_val = float(phi_ev.mean() + u_ev.mean())
            mart  = float(np.abs(s2star_ev - s1_eval).mean())

            history["it"].append(it)
            history["L"].append(L_val)
            history["mart_err"].append(mart)
            print(f"  {it:>6d} | L = {L_val:+.4f} | |mart| = {mart:.4f}")

    # --- Restore best model ---
    if best_u_w is not None:
        u_net.set_weights(best_u_w)
        h_net.set_weights(best_h_w)
        print(f"\n[✓] Best model restored  (min martingale error = {best_mart:.4f})")

    return u_net, h_net, history


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_results(
    u_net: MLP,
    h_net: MLP,
    history: dict,
    sample_mu1,
    sample_mu2,
    cost_fn,
    *,
    title: str = "MOT Semi-Dual",
    exact: float | None = None,
    save_path: str | None = None,
):
    """
    Three-panel summary plot:
      (1) convergence of the dual objective L(u, h)
      (2) reconstructed transport plan (scatter S1 → T(S1))

    Parameters
    ----------
    exact     : known exact value to overlay as a reference line
    save_path : if given, save the figure to this path
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # --- (1) Convergence ---
    ax = axes[0]
    ax.plot(history["it"], history["L"], "#1f77b4", lw=1.5, label="Dual L(u,h)")
    if exact is not None:
        ax.axhline(exact, color="red", ls="--", lw=2, label=f"Exact = {exact}")
    last = float(np.mean(history["L"][-5:]))
    ax.axhline(last, color="green", ls=":", lw=1.5, label=f"Final ≈ {last:.4f}")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("$L(u, h)$")
    ax.set_title("Convergence", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # --- (2) Transport plan ---
    ax = axes[1]
    np.random.seed(777)
    s1t  = sample_mu1(3000)
    s2p  = sample_mu2(5000)
    _, s2star = mart_c_transform(u_net, h_net, cost_fn, s1t, s2p)
    ax.scatter(s1t.flatten(), s2star.flatten(), s=3, alpha=0.2, c="#1f77b4")
    lims = [
        min(s1t.min(), s2star.min()),
        max(s1t.max(), s2star.max()),
    ]
    ax.plot(lims, lims, "r--", lw=1.5, alpha=0.5, label="$s_2 = s_1$")
    ax.set_xlabel("$S_1$")
    ax.set_ylabel("$T(S_1)$")
    ax.set_title("Transport plan", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Figure saved → {save_path}")

    plt.show()

"""
run_sinkhorn.py
---------------
Example script: run the entropic MOT Sinkhorn solver on a chosen payoff
and marginal pair, then display the results.

Usage
-----
    python run_sinkhorn.py                           # defaults
    python run_sinkhorn.py --cost asian --eps 0.01 --n 150
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from mot import COST_REGISTRY, DISTRIBUTION_PAIRS, sinkhorn


def build_payoff_matrix(cost_fn, x_atoms, y_atoms):
    """Evaluate cost_fn on the full grid to get the payoff matrix (m1, m2)."""
    return cost_fn(x_atoms[:, None], y_atoms[None, :])


def main():
    parser = argparse.ArgumentParser(description="MOT Sinkhorn (entropic) solver")
    parser.add_argument("--cost", default="asian",
                        choices=list(COST_REGISTRY.keys()),
                        help="Payoff function (default: asian)")
    parser.add_argument("--dist", default="uniform",
                        choices=list(DISTRIBUTION_PAIRS.keys()),
                        help="Marginal distribution pair (default: uniform)")
    parser.add_argument("--eps",  type=float, default=0.005,
                        help="Entropic regularisation ε (default: 0.005)")
    parser.add_argument("--n",    type=int, default=100,
                        help="Grid size (atoms per marginal, default: 100)")
    parser.add_argument("--n_iter", type=int, default=1000,
                        help="Max Sinkhorn iterations (default: 1000)")
    parser.add_argument("--save", type=str, default=None,
                        help="Path to save the output figure")
    args = parser.parse_args()

    cost_fn             = COST_REGISTRY[args.cost]
    dist1, dist2, label = DISTRIBUTION_PAIRS[args.dist]

    # Determine support bounds from the 0.001 / 0.999 quantiles
    a1, b1 = dist1.ppf(1e-4), dist1.ppf(1 - 1e-4)
    a2, b2 = dist2.ppf(1e-4), dist2.ppf(1 - 1e-4)

    x_atoms, alpha = sinkhorn.discretize(dist1, a1, b1, args.n)
    y_atoms, beta  = sinkhorn.discretize(dist2, a2, b2, args.n)
    G_mat          = build_payoff_matrix(cost_fn, x_atoms, y_atoms)

    print(f"\n{'='*60}")
    print(f"  Payoff      : {args.cost}")
    print(f"  Marginals   : {label}")
    print(f"  Grid size   : {args.n}")
    print(f"  ε           : {args.eps}")
    print(f"{'='*60}\n")

    P, val, history, u1, u2, h = sinkhorn.solve(
        x_atoms, alpha, y_atoms, beta, G_mat,
        eps=args.eps, n_iter=args.n_iter,
    )

    print(f"\n  Estimated robust price (E[c]) : {val:.4f}")

    title = f"Sinkhorn MOT  |  cost={args.cost}  |  {label}"
    sinkhorn.plot_results(
        x_atoms, y_atoms, P, history, args.eps,
        title=title,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()

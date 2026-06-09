"""
run_semi_dual.py
----------------
Example script: run the MOT semi-dual solver on a chosen payoff and
marginal pair, then display the convergence and transport plan.

Usage
-----
    python run_semi_dual.py                        # defaults
    python run_semi_dual.py --cost max --dist uniform --n_iter 5000
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from mot import COST_REGISTRY, DISTRIBUTION_PAIRS, semi_dual


def main():
    parser = argparse.ArgumentParser(description="MOT semi-dual solver")
    parser.add_argument("--cost",    default="max",
                        choices=list(COST_REGISTRY.keys()),
                        help="Payoff function (default: max)")
    parser.add_argument("--dist",    default="uniform",
                        choices=list(DISTRIBUTION_PAIRS.keys()),
                        help="Marginal distribution pair (default: uniform)")
    parser.add_argument("--n_iter",  type=int, default=5000,
                        help="Training iterations (default: 5000)")
    parser.add_argument("--lr",      type=float, default=5e-4,
                        help="Adam learning rate for both networks (default: 5e-4)")
    parser.add_argument("--hidden",  type=int, default=32,
                        help="Hidden layer width (default: 32)")
    parser.add_argument("--exact",   type=float, default=None,
                        help="Known exact value for convergence plot")
    parser.add_argument("--save",    type=str, default=None,
                        help="Path to save the output figure (e.g. results/plot.png)")
    args = parser.parse_args()

    cost_fn              = COST_REGISTRY[args.cost]
    dist1, dist2, label  = DISTRIBUTION_PAIRS[args.dist]

    mu1 = lambda n: dist1.rvs(size=(n, 1))
    mu2 = lambda n: dist2.rvs(size=(n, 1))

    print(f"\n{'='*60}")
    print(f"  Payoff      : {args.cost}")
    print(f"  Marginals   : {label}")
    print(f"  Iterations  : {args.n_iter}")
    print(f"{'='*60}\n")

    u_net, h_net, history = semi_dual.solve(
        mu1, mu2, cost_fn,
        n_iter=args.n_iter,
        lr_u=args.lr,
        lr_h=args.lr,
        hidden_size=args.hidden,
    )

    final_price = float(np.mean(history["L"][-5:]))
    print(f"\n  Estimated robust price : {final_price:.4f}")

    title = f"MOT Semi-Dual  |  cost={args.cost}  |  {label}"
    semi_dual.plot_results(
        u_net, h_net, history, mu1, mu2, cost_fn,
        title=title,
        exact=args.exact,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()

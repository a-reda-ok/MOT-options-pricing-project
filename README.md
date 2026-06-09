# Exotic Options Pricing via Martingale Optimal Transport

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)]()
[![NumPy](https://img.shields.io/badge/numpy-1.24%2B-orange)]()
[![SciPy](https://img.shields.io/badge/scipy-1.10%2B-green)]()

A clean NumPy/SciPy implementation of three numerical methods for computing
**model-independent (robust) upper bounds** on exotic option prices using
**Martingale Optimal Transport (MOT)**.

> Project report: *Exotic Options Pricing by Using Martingale Optimal Transport*  
> CentraleSupélec — Supervisor: Guo Gaoyue — April 2026

---

## Background

Given marginal distributions µ₁ and µ₂ of an asset at two dates (implied by
observable call prices), the **robust pricing problem** is:

```
MK_c(µ₁, µ₂) = sup_{P ∈ M(µ₁, µ₂)}  E^P[ c(S₁, S₂) ]
```

where `M(µ₁, µ₂)` is the set of martingale couplings (joint laws consistent
with no-arbitrage).  By the **Beiglböck–Henry-Labordère–Penkner duality
theorem (2013)**, this equals the cheapest semi-static super-replication cost.

Three numerical methods are implemented:

| Method | File | Key idea |
|--------|------|----------|
| **Semi-dual** | `mot/semi_dual.py` | Partial dualization → convex min; dual potentials *u*, *h* parametrised by two MLPs; gradients via Danskin's theorem |
| **Sinkhorn (entropic)** | `mot/sinkhorn.py` | Add entropic regularization ε H(P\|ρ₀); alternating updates of *u₁*, *u₂*, *h* in log-space |
| *(LP — notebook only)* | `notebooks/` | Direct discretization as a linear program (benchmark, does not scale) |

---

## Repository structure

```
mot_pricing/
├── mot/                        # Core library
│   ├── __init__.py
│   ├── cost_functions.py       # Exotic option payoffs
│   ├── distributions.py        # Marginal distributions + convex-order check
│   ├── neural_net.py           # Two-layer MLP with Adam (used by semi-dual)
│   ├── semi_dual.py            # Semi-dual solver + visualisation
│   └── sinkhorn.py             # Entropic Sinkhorn solver + visualisation
│
├── notebooks/
│   └── sinkhorn_exploration.ipynb   # Step-by-step Sinkhorn walkthrough
│
├── run_semi_dual.py            # CLI entry-point for the semi-dual method
├── run_sinkhorn.py             # CLI entry-point for the Sinkhorn method
├── requirements.txt
└── README.md
```

---

## Quickstart

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the semi-dual solver

```bash
# Lookback option on U[1,3] → U[0,4]  (5 000 iterations)
python run_semi_dual.py --cost max --dist uniform --n_iter 5000

# Asian option, save figure
python run_semi_dual.py --cost asian --dist uniform --n_iter 5000 --save results/asian_semidual.png

# All options
python run_semi_dual.py --help
```

### Run the Sinkhorn (entropic) solver

```bash
# Asian option, ε = 0.005, 100-point grid
python run_sinkhorn.py --cost asian --eps 0.005 --n 100

# Lookback option, save figure
python run_sinkhorn.py --cost max --eps 0.005 --save results/max_sinkhorn.png

# All options
python run_sinkhorn.py --help
```

### Use the library directly

```python
import numpy as np
from mot import cost_asian, uniform_S1, uniform_S2, semi_dual

mu1 = lambda n: uniform_S1.rvs(size=(n, 1))
mu2 = lambda n: uniform_S2.rvs(size=(n, 1))

u_net, h_net, history = semi_dual.solve(
    mu1, mu2, cost_asian,
    n_iter=5000, lr_u=5e-4, lr_h=5e-4,
)

price = float(np.mean(history["L"][-5:]))
print(f"Robust price (Asian): {price:.4f}")   # ≈ 0.66
```

---

## Available payoffs

| Key | Formula | Description |
|-----|---------|-------------|
| `call` | `(S₂ − K)⁺` | European call on S₂ |
| `abs` | `\|S₂ − S₁\|` | Absolute price change |
| `asian` | `((S₁+S₂)/2 − K)⁺` | Arithmetic-average Asian call |
| `max` | `(max(S₁,S₂) − K)⁺` | Lookback / max call |
| `barrier` | `(max(S₁,S₂)−K)⁺ · 1{max≤B}` | Up-and-out barrier |
| `cubic` | `(S₁+S₂)³` | Cubic cost (stress test) |

Default strike `K = 1.5`, barrier `B = 3.0`.

## Available marginal pairs

| Key | µ₁ | µ₂ | Convex order |
|-----|----|----|-------------|
| `uniform` | U[1, 3] | U[0, 4] | ✓ same mean, Var₂ > Var₁ |
| `gamma` | Γ(10, 0.1) | Γ(2, 0.5) | ✓ same mean |
| `lognorm` | LogN(σ=0.2) | LogN(σ=0.4) | ✓ same mean (≈1) |

---

## Numerical results (benchmark)

All prices computed on `uniform` marginals (S₁ ∼ U[1,3], S₂ ∼ U[0,4]):

| Payoff | LP (exact) | Semi-dual | Sinkhorn (ε=0.005) |
|--------|-----------|-----------|-------------------|
| `abs` | 1.13 | 0.99 | 0.99 |
| `asian` | 0.66 | 0.66 | 0.65 |
| `max` | 1.09 | 1.04 | 1.02 |

The semi-dual method slightly underestimates the LP price because the
deterministic argmax recovers only one branch of the optimal two-map
transport (Beiglböck-Juillet theorem, 2016).

---

## Method summary

### Semi-dual (neural network)

The Kantorovich dual eliminates the µ₁ constraint:

```
MK_c = inf_{u, h}  E^µ2[u(S₂)] + E^µ1[ sup_{s₂} { c(S₁,s₂) − u(s₂) − h(S₁)(s₂−S₁) } ]
```

- **Convex** in (u, h) — no saddle-point instability.
- Gradients via **Danskin's theorem**: differentiate through the argmax for free.
- Both u and h parametrised by 2-hidden-layer MLPs (width 32, Softplus activation).
- Updated jointly with **Adam** (lr = 5×10⁻⁴, gradient clip = 5).
- Best model restored from the last 1 000 iterations (minimum martingale error).

### Sinkhorn (entropic relaxation)

The regularised primal:

```
MK_ε = sup_{P ∈ M(µ₁,µ₂)}  E^P[c] − ε H(P | ρ₀)
```

Dual updates alternate:
1. **u₁** — closed-form log-sum-exp (µ₁ marginal constraint)
2. **h** — scalar root-finding per row (martingale constraint, Brent's method)
3. **u₂** — closed-form log-sum-exp (µ₂ marginal constraint)

All exponentials computed in log-space (log-sum-exp trick) for stability.
Converges in O(100) iterations for ε = 0.005.

---

## References

1. Beiglböck, Henry-Labordère, Penkner (2013). *Model-independent bounds for option prices.* Finance & Stochastics.
2. Henry-Labordère (2019). *(Martingale) Optimal Transport and Anomaly Detection with Neural Networks.* arXiv:1904.04546.
3. Beiglböck & Juillet (2016). *On a problem of optimal transport under marginal martingale constraints.* Annals of Probability.
4. Villani (2003). *Topics in Optimal Transportation.* AMS.
5. Danskin (1967). *The Theory of Max-Min.*
6. Kingma & Ba (2015). *Adam.* arXiv:1412.6980.
7. Cuturi (2013). *Sinkhorn Distances.* NeurIPS.
8. Strassen (1965). *The existence of probability measures with given marginals.* Ann. Math. Stat.
9. Boyd & Vandenberghe (2004). *Convex Optimization.* Cambridge.

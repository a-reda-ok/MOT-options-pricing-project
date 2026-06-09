"""
mot — Martingale Optimal Transport pricing toolkit.

Modules
-------
cost_functions  : exotic option payoff functions
distributions   : marginal distributions (uniform, gamma, lognormal)
neural_net      : MLP used in the semi-dual solver
semi_dual       : semi-dual MOT solver (neural network parametrisation)
sinkhorn        : entropic MOT solver (Sinkhorn algorithm)
"""

from .cost_functions import (
    cost_call,
    cost_abs,
    cost_asian,
    cost_max,
    cost_barrier,
    cost_cubic,
    COST_REGISTRY,
)
from .distributions import (
    uniform_S1, uniform_S2,
    gamma_S1,   gamma_S2,
    lognorm_S1, lognorm_S2,
    DISTRIBUTION_PAIRS,
    check_convex_order,
)
from . import semi_dual, sinkhorn

__all__ = [
    "cost_call", "cost_abs", "cost_asian", "cost_max", "cost_barrier", "cost_cubic",
    "COST_REGISTRY",
    "uniform_S1", "uniform_S2", "gamma_S1", "gamma_S2",
    "lognorm_S1", "lognorm_S2", "DISTRIBUTION_PAIRS", "check_convex_order",
    "semi_dual", "sinkhorn",
]

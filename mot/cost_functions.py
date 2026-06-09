"""
cost_functions.py
-----------------
Payoff functions for exotic options used in the MOT pricing framework.

Each function takes arrays s1, s2 (asset prices at t=1 and t=2) and returns
the payoff c(S1, S2).  All functions are vectorised over NumPy arrays.

References
----------
[1] Beiglböck, Henry-Labordère, Penkner (2013) — MOT framework.
[14] Henry-Labordère (2017) — Model-free Hedging.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Vanilla helper
# ---------------------------------------------------------------------------

def _to_array(*args):
    return [np.asarray(a, dtype=float) for a in args]


# ---------------------------------------------------------------------------
# Payoff functions
# ---------------------------------------------------------------------------

def cost_call(s1, s2, K: float = 1.5):
    """European call on S2:  (S2 - K)+."""
    s1, s2 = _to_array(s1, s2)
    return np.maximum(s2 - K, 0.0)


def cost_abs(s1, s2):
    """Absolute difference:  |S2 - S1|."""
    s1, s2 = _to_array(s1, s2)
    return np.abs(s1 - s2)


def cost_asian(s1, s2, K: float = 1.5):
    """Asian call on the arithmetic average:  ((S1+S2)/2 - K)+."""
    s1, s2 = _to_array(s1, s2)
    return np.maximum((s1 + s2) / 2.0 - K, 0.0)


def cost_max(s1, s2, K: float = 1.5):
    """Lookback / max call:  (max(S1, S2) - K)+."""
    s1, s2 = _to_array(s1, s2)
    return np.maximum(np.maximum(s1, s2) - K, 0.0)


def cost_barrier(s1, s2, K: float = 1.5, B: float = 3.0):
    """Up-and-out barrier option:  (max(S1,S2)-K)+ * 1{max(S1,S2) <= B}."""
    s1, s2 = _to_array(s1, s2)
    max_s = np.maximum(s1, s2)
    return np.where(max_s > B, 0.0, np.maximum(max_s - K, 0.0))


def cost_cubic(s1, s2):
    """Cubic cost:  (S1 + S2)^3  (used to stress-test convexity)."""
    s1, s2 = _to_array(s1, s2)
    return (s1 + s2) ** 3


# ---------------------------------------------------------------------------
# Registry  (name → callable) for convenient selection
# ---------------------------------------------------------------------------

COST_REGISTRY = {
    "call":    cost_call,
    "abs":     cost_abs,
    "asian":   cost_asian,
    "max":     cost_max,
    "barrier": cost_barrier,
    "cubic":   cost_cubic,
}

"""
distributions.py
----------------
Marginal distributions for S1 and S2 used in the MOT pricing experiments.

Each entry is a scipy.stats frozen distribution.  A convex-order check
(Strassen's theorem) is provided as a utility.

References
----------
[8] Strassen (1965) — existence of martingale couplings iff µ1 ≤_cx µ2.
"""

import numpy as np
import scipy.stats as stats


# ---------------------------------------------------------------------------
# Uniform marginals  (main benchmark)
# ---------------------------------------------------------------------------

uniform_S1 = stats.uniform(loc=1, scale=2)   # U[1, 3]
uniform_S2 = stats.uniform(loc=0, scale=4)   # U[0, 4]

# ---------------------------------------------------------------------------
# Gamma marginals
# ---------------------------------------------------------------------------

gamma_S1 = stats.gamma(a=10.0, scale=0.1)    # mean=1, small variance
gamma_S2 = stats.gamma(a=2.0,  scale=0.5)    # mean=1, larger variance

# ---------------------------------------------------------------------------
# Log-normal marginals  (financial Black-Scholes-type)
# ---------------------------------------------------------------------------

_vol1 = 0.2
_vol2 = 0.4

lognorm_S1 = stats.lognorm(s=_vol1, scale=np.exp(-0.5 * _vol1 ** 2))  # low vol
lognorm_S2 = stats.lognorm(s=_vol2, scale=np.exp(-0.5 * _vol2 ** 2))  # high vol

# ---------------------------------------------------------------------------
# Named pairs for convenient experiment selection
# ---------------------------------------------------------------------------

DISTRIBUTION_PAIRS = {
    "uniform": (uniform_S1, uniform_S2, "U[1,3] → U[0,4]"),
    "gamma":   (gamma_S1,   gamma_S2,   "Gamma(10,0.1) → Gamma(2,0.5)"),
    "lognorm": (lognorm_S1, lognorm_S2, f"LogNorm(σ={_vol1}) → LogNorm(σ={_vol2})"),
}


# ---------------------------------------------------------------------------
# Utility: convex-order check (Strassen's theorem, necessary condition)
# ---------------------------------------------------------------------------

def check_convex_order(dist1, dist2, n_mc: int = 200_000, seed: int = 0) -> dict:
    """
    Monte-Carlo check of the convex order condition µ1 ≤_cx µ2.

    Parameters
    ----------
    dist1, dist2 : scipy.stats frozen distributions
    n_mc         : number of Monte-Carlo samples
    seed         : random seed

    Returns
    -------
    dict with keys 'mean_ok', 'variance_ok', 'mean1', 'mean2', 'var1', 'var2'
    """
    rng = np.random.default_rng(seed)
    s1 = dist1.rvs(size=n_mc, random_state=rng)
    s2 = dist2.rvs(size=n_mc, random_state=rng)

    m1, m2 = s1.mean(), s2.mean()
    v1, v2 = s1.var(),  s2.var()

    return {
        "mean_ok":    bool(np.isclose(m1, m2, rtol=1e-2)),
        "variance_ok": bool(v1 <= v2 + 1e-6),
        "mean1": m1, "mean2": m2,
        "var1":  v1, "var2":  v2,
    }

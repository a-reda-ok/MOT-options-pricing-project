"""
neural_net.py
-------------
Lightweight two-hidden-layer feedforward network (NumPy only) used to
parametrise the dual potentials u and h in the MOT semi-dual formulation.

Architecture:  input(1) → H → H → output(1)
Activation:    Softplus  σ(x) = log(1 + e^x)
Optimiser:     Adam [6] with gradient clipping

References
----------
[5]  Danskin (1967)         — gradient computation via argmax.
[6]  Kingma & Ba (2015)     — Adam optimiser.
[9]  Cybenko (1989)         — universal approximation theorem.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Activation function and its derivative
# ---------------------------------------------------------------------------

def softplus(x: np.ndarray) -> np.ndarray:
    """Numerically stable softplus: log(1 + e^x)."""
    return np.where(x > 20, x, np.log1p(np.exp(np.clip(x, -50, 20))))


def softplus_grad(x: np.ndarray) -> np.ndarray:
    """Derivative of softplus = sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------

class MLP:
    """
    Two-hidden-layer MLP with Softplus activations.

    Parameters
    ----------
    hidden_size : int
        Width of each hidden layer (default: 32).
    """

    def __init__(self, hidden_size: int = 32):
        H = hidden_size
        sc = lambda a, b: np.sqrt(2.0 / (a + b))  # He-style init

        self.W1 = np.random.randn(H, 1) * sc(1, H)
        self.b1 = np.zeros(H)
        self.W2 = np.random.randn(H, H) * sc(H, H)
        self.b2 = np.zeros(H)
        self.W3 = np.random.randn(1, H) * sc(H, 1)
        self.b3 = np.zeros(1)

        self._param_names = ["W1", "b1", "W2", "b2", "W3", "b3"]
        self._cache: dict = {}
        self.grads: dict = {}

        # Adam state
        self._t = 0
        self._m = {p: np.zeros_like(getattr(self, p)) for p in self._param_names}
        self._v = {p: np.zeros_like(getattr(self, p)) for p in self._param_names}

    # ------------------------------------------------------------------
    # Forward / backward
    # ------------------------------------------------------------------

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass.  x: (N, 1) → out: (N, 1)."""
        z1 = x @ self.W1.T + self.b1
        a1 = softplus(z1)
        z2 = a1 @ self.W2.T + self.b2
        a2 = softplus(z2)
        out = a2 @ self.W3.T + self.b3
        self._cache = {"x": x, "z1": z1, "a1": a1, "z2": z2, "a2": a2}
        return out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """
        Backpropagation.  d_out: (N, 1) upstream gradient → dx: (N, 1).
        Also populates self.grads with parameter gradients.
        """
        x  = self._cache["x"]
        z1 = self._cache["z1"]
        a1 = self._cache["a1"]
        z2 = self._cache["z2"]
        a2 = self._cache["a2"]
        n  = x.shape[0]

        # Layer 3
        dW3 = d_out.T @ a2 / n
        db3 = d_out.mean(0)
        da2 = d_out @ self.W3

        # Layer 2
        dz2 = da2 * softplus_grad(z2)
        dW2 = dz2.T @ a1 / n
        db2 = dz2.mean(0)
        da1 = dz2 @ self.W2

        # Layer 1
        dz1 = da1 * softplus_grad(z1)
        dW1 = dz1.T @ x / n
        db1 = dz1.mean(0)
        dx  = dz1 @ self.W1

        self.grads = {
            "W1": dW1, "b1": db1,
            "W2": dW2, "b2": db2,
            "W3": dW3, "b3": db3,
        }
        return dx

    # ------------------------------------------------------------------
    # Optimiser step
    # ------------------------------------------------------------------

    def step(self, lr: float, sign: float = 1.0, clip: float = 5.0):
        """
        Adam update [6].

        Parameters
        ----------
        lr   : learning rate
        sign : +1.0 for gradient descent (minimisation),
               -1.0 for gradient ascent (maximisation).
        clip : gradient clipping threshold.
        """
        self._t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for p in self._param_names:
            g = np.clip(self.grads[p], -clip, clip)
            self._m[p] = beta1 * self._m[p] + (1 - beta1) * g
            self._v[p] = beta2 * self._v[p] + (1 - beta2) * g ** 2
            m_hat = self._m[p] / (1 - beta1 ** self._t)
            v_hat = self._v[p] / (1 - beta2 ** self._t)
            current = getattr(self, p)
            setattr(self, p, current - sign * lr * m_hat / (np.sqrt(v_hat) + eps))

    # ------------------------------------------------------------------
    # Weight snapshot helpers (for best-model restoration)
    # ------------------------------------------------------------------

    def get_weights(self) -> dict:
        return {p: getattr(self, p).copy() for p in self._param_names}

    def set_weights(self, weights: dict):
        for p, v in weights.items():
            setattr(self, p, v.copy())

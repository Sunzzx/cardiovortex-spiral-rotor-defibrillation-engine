import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class APParameters:
    """Parameters for Aliev-Panfilov (1996) 2-variable cardiac model."""
    k: float = 8.0         # Reaction rate scaling constant
    a: float = 0.15        # Threshold parameter for depolarization
    eps0: float = 0.002    # Baseline recovery rate
    mu1: float = 0.2       # Recovery dynamic coefficient 1
    mu2: float = 0.3       # Recovery dynamic coefficient 2
    u_rest: float = 0.0    # Normalized resting potential
    u_peak: float = 1.0    # Normalized peak action potential

class AlievPanfilovModel:
    """
    Aliev-Panfilov (1996) model for cardiac tissue excitation.
    
    Equations:
        du/dt = -k * u * (u - a) * (u - 1) - u * v + I_ext
        dv/dt = eps(u, v) * (-v - k * u * (u - a - 1))
        eps(u, v) = eps0 + (mu1 * v) / (u + mu2)
    """
    def __init__(self, params: APParameters = None):
        self.params = params or APParameters()

    def reaction_rates(
        self,
        u: np.ndarray,
        v: np.ndarray,
        i_ext: np.ndarray = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute du/dt and dv/dt at current state."""
        p = self.params
        # du/dt
        du = -p.k * u * (u - p.a) * (u - 1.0) - u * v + i_ext
        
        # eps(u, v)
        denom = np.maximum(u + p.mu2, 1e-6)
        eps = p.eps0 + (p.mu1 * v) / denom
        
        # dv/dt
        dv = eps * (-v - p.k * u * (u - p.a - 1.0))
        return du, dv

    def initial_state(self, shape: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray]:
        """Resting initial state across given grid shape."""
        u = np.full(shape, self.params.u_rest, dtype=np.float64)
        v = np.zeros(shape, dtype=np.float64)
        return u, v

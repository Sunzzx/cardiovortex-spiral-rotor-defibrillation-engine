import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class FHNParameters:
    """FitzHugh-Nagumo excitable medium model parameters."""
    a: float = 0.7
    b: float = 0.8
    eps: float = 0.08
    i_bias: float = 0.0

class FitzHughNagumoModel:
    """
    FitzHugh-Nagumo reaction model:
        du/dt = u - (u^3)/3 - v + I_ext
        dv/dt = eps * (u + a - b * v)
    """
    def __init__(self, params: FHNParameters = None):
        self.params = params or FHNParameters()

    def reaction_rates(
        self,
        u: np.ndarray,
        v: np.ndarray,
        i_ext: np.ndarray = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        p = self.params
        du = u - (u**3) / 3.0 - v + p.i_bias + i_ext
        dv = p.eps * (u + p.a - p.b * v)
        return du, dv

    def initial_state(self, shape: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray]:
        u = np.full(shape, -1.199, dtype=np.float64)
        v = np.full(shape, -0.624, dtype=np.float64)
        return u, v

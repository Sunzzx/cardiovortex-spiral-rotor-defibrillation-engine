import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class RotorCore:
    """Coordinates and topological property of a detected cardiac rotor core."""
    x: float
    y: float
    charge: int         # +1 (clockwise) or -1 (counter-clockwise)
    phase: float        # Local phase angle [-pi, pi]
    time: float

class PhaseSingularityDetector:
    """
    Topological phase singularity and rotor core detection via winding number.
    
    Given state (u, v), the phase is defined as:
        theta(x, y) = atan2(v - v_star, u - u_star)
    The topological charge Q along an elementary cell boundary loop is:
        Q = (1 / 2pi) * Sum wrap(Delta theta_k) in {-1, +1}
    """
    def __init__(self, u_star: float = 0.3, v_star: float = 0.2):
        self.u_star = u_star
        self.v_star = v_star

    def compute_phase(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Compute continuous phase angle theta in [-pi, pi]."""
        return np.arctan2(v - self.v_star, u - self.u_star)

    @staticmethod
    def wrap_angle(delta: np.ndarray) -> np.ndarray:
        """Wrap angular difference into [-pi, pi)."""
        return (delta + np.pi) % (2.0 * np.pi) - np.pi

    def detect(self, u: np.ndarray, v: np.ndarray, time: float = 0.0) -> List[RotorCore]:
        """
        Detect all phase singularities (rotor cores) on the tissue grid.
        """
        phase = self.compute_phase(u, v)
        ny, nx = phase.shape
        cores: List[RotorCore] = []
        
        # Consider 2x2 elementary square loops
        # Path: (i, j) -> (i, j+1) -> (i+1, j+1) -> (i+1, j) -> (i, j)
        p00 = phase[:-1, :-1]
        p01 = phase[:-1, 1:]
        p11 = phase[1:, 1:]
        p10 = phase[1:, :-1]
        
        d0 = self.wrap_angle(p01 - p00)
        d1 = self.wrap_angle(p11 - p01)
        d2 = self.wrap_angle(p10 - p11)
        d3 = self.wrap_angle(p00 - p10)
        
        loop_integral = d0 + d1 + d2 + d3
        charges = np.round(loop_integral / (2.0 * np.pi)).astype(int)
        
        # Find locations where charge != 0
        singularity_indices = np.argwhere(charges != 0)
        
        for y_idx, x_idx in singularity_indices:
            q = int(charges[y_idx, x_idx])
            if abs(q) >= 1:
                # Normalizing charge to +1 or -1
                charge_val = 1 if q > 0 else -1
                core = RotorCore(
                    x=float(x_idx + 0.5),
                    y=float(y_idx + 0.5),
                    charge=charge_val,
                    phase=float(p00[y_idx, x_idx]),
                    time=time,
                )
                cores.append(core)
                
        return cores

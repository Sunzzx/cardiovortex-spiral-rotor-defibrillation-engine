import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Callable
from cardiovortex.models.aliev_panfilov import AlievPanfilovModel

class BoundaryCondition(str, Enum):
    NEUMANN = "neumann"      # No-flux (isolated tissue)
    PERIODIC = "periodic"    # Toroidal / continuous boundary
    DIRICHLET = "dirichlet"  # Fixed boundary potential

@dataclass
class TissueGrid:
    """2D anisotropic myocardial tissue grid definition."""
    nx: int = 80
    ny: int = 80
    dx: float = 0.02          # cm
    dy: float = 0.02          # cm
    d_longitudinal: float = 1.0e-3  # cm^2 / ms (along fiber direction)
    d_transverse: float = 2.5e-4    # cm^2 / ms (across fibers, ~4:1 anisotropy)
    fiber_angle: float = 0.0        # radians (rotation angle of myocardial fibers)

    @property
    def diffusion_tensor(self) -> Tuple[float, float, float]:
        """Compute rotated diffusion components D_xx, D_yy, D_xy."""
        cos_t = np.cos(self.fiber_angle)
        sin_t = np.sin(self.fiber_angle)
        d_par = self.d_longitudinal
        d_perp = self.d_transverse
        
        d_xx = d_par * (cos_t ** 2) + d_perp * (sin_t ** 2)
        d_yy = d_par * (sin_t ** 2) + d_perp * (cos_t ** 2)
        d_xy = (d_par - d_perp) * cos_t * sin_t
        return d_xx, d_yy, d_xy

class CardiacSimulator:
    """
    Finite-difference reaction-diffusion solver for cardiac excitable media.
    
    Solves:
        du/dt = Div(D Grad u) + f(u, v) + I_ext
        dv/dt = g(u, v)
    """
    def __init__(
        self,
        model: Optional[AlievPanfilovModel] = None,
        grid: Optional[TissueGrid] = None,
        boundary: BoundaryCondition = BoundaryCondition.NEUMANN,
    ):
        self.model = model or AlievPanfilovModel()
        self.grid = grid or TissueGrid()
        self.boundary = boundary
        
        # State variables
        self.u, self.v = self.model.initial_state((self.grid.ny, self.grid.nx))
        self.time: float = 0.0
        self.step_count: int = 0

    def check_cfl_condition(self, dt: float) -> bool:
        """Verify Courant-Friedrichs-Lewy parabolic stability criterion."""
        d_xx, d_yy, _ = self.grid.diffusion_tensor
        max_d = max(d_xx, d_yy)
        min_dx = min(self.grid.dx, self.grid.dy)
        cfl_dt_max = (min_dx ** 2) / (4.0 * max_d)
        return bool(dt <= cfl_dt_max)

    def compute_laplacian(self, field: np.ndarray) -> np.ndarray:
        """Compute anisotropic Laplacian with Neumann or Periodic boundaries."""
        d_xx, d_yy, d_xy = self.grid.diffusion_tensor
        dx, dy = self.grid.dx, self.grid.dy
        ny, nx = self.grid.ny, self.grid.nx
        
        if self.boundary == BoundaryCondition.NEUMANN:
            # Pad field with no-flux Neumann conditions: du/dn = 0
            # u[-1] = u[1], u[N] = u[N-2]
            padded = np.pad(field, 1, mode="edge")
            
            # Second derivatives
            d2_dx2 = (padded[1:-1, 2:] - 2.0 * field + padded[1:-1, :-2]) / (dx ** 2)
            d2_dy2 = (padded[2:, 1:-1] - 2.0 * field + padded[:-2, 1:-1]) / (dy ** 2)
            
            lap = d_xx * d2_dx2 + d_yy * d2_dy2
            
            # Cross-derivative if fibers are rotated
            if abs(d_xy) > 1e-12:
                d2_dxdy = (
                    padded[2:, 2:] - padded[2:, :-2] - padded[:-2, 2:] + padded[:-2, :-2]
                ) / (4.0 * dx * dy)
                lap += 2.0 * d_xy * d2_dxdy
            return lap
            
        elif self.boundary == BoundaryCondition.PERIODIC:
            # Periodic wrapping
            u_left = np.roll(field, 1, axis=1)
            u_right = np.roll(field, -1, axis=1)
            u_up = np.roll(field, 1, axis=0)
            u_down = np.roll(field, -1, axis=0)
            
            d2_dx2 = (u_right - 2.0 * field + u_left) / (dx ** 2)
            d2_dy2 = (u_down - 2.0 * field + u_up) / (dy ** 2)
            lap = d_xx * d2_dx2 + d_yy * d2_dy2
            
            if abs(d_xy) > 1e-12:
                u_ne = np.roll(np.roll(field, -1, axis=0), -1, axis=1)
                u_nw = np.roll(np.roll(field, -1, axis=0), 1, axis=1)
                u_se = np.roll(np.roll(field, 1, axis=0), -1, axis=1)
                u_sw = np.roll(np.roll(field, 1, axis=0), 1, axis=1)
                d2_dxdy = (u_ne - u_nw - u_se + u_sw) / (4.0 * dx * dy)
                lap += 2.0 * d_xy * d2_dxdy
            return lap
            
        else:  # Dirichlet (zero on boundary)
            padded = np.pad(field, 1, mode="constant", constant_values=0.0)
            d2_dx2 = (padded[1:-1, 2:] - 2.0 * field + padded[1:-1, :-2]) / (dx ** 2)
            d2_dy2 = (padded[2:, 1:-1] - 2.0 * field + padded[:-2, 1:-1]) / (dy ** 2)
            return d_xx * d2_dx2 + d_yy * d2_dy2

    def step(self, dt: float, i_ext: Optional[np.ndarray] = None):
        """Advance the state by dt using forward Euler time-integration."""
        if i_ext is None:
            i_ext = 0.0
            
        # Reaction kinetics
        du_react, dv_react = self.model.reaction_rates(self.u, self.v, i_ext)
        
        # Spatial electrical diffusion
        diffusion_u = self.compute_laplacian(self.u)
        
        # Update states
        self.u += dt * (diffusion_u + du_react)
        self.v += dt * dv_react
        
        # Keep numerical stability in non-linear regime
        np.clip(self.u, -0.5, 2.5, out=self.u)
        np.clip(self.v, -0.5, 2.5, out=self.v)
        
        self.time += dt
        self.step_count += 1

    def apply_stimulus(
        self,
        x_range: Tuple[int, int],
        y_range: Tuple[int, int],
        amplitude: float = 1.0,
    ):
        """Apply localized external current depolarizing specified region."""
        x0, x1 = x_range
        y0, y1 = y_range
        self.u[y0:y1, x0:x1] = amplitude

    def initialize_spiral_wave(
        self,
        s1_amplitude: float = 1.0,
        s2_amplitude: float = 1.0,
    ):
        """
        Classic cross-field (S1-S2) stimulation protocol.
        Creates an initial broken wave that curls into a stable rotating spiral wave (rotor).
        """
        ny, nx = self.grid.ny, self.grid.nx
        # S1: planar wave across left half
        self.u[:, : nx // 2] = s1_amplitude
        # S2: refractory tail across upper quadrant
        self.v[: ny // 2, :] = 0.8

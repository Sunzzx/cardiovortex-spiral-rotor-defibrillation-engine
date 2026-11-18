import numpy as np
import pytest
from cardiovortex.models.aliev_panfilov import AlievPanfilovModel
from cardiovortex.simulator import CardiacSimulator, TissueGrid, BoundaryCondition

def test_diffusion_tensor_isotropic():
    grid = TissueGrid(d_longitudinal=1e-3, d_transverse=1e-3, fiber_angle=np.pi / 4)
    dxx, dyy, dxy = grid.diffusion_tensor
    assert np.isclose(dxx, 1e-3)
    assert np.isclose(dyy, 1e-3)
    assert np.isclose(dxy, 0.0, atol=1e-12)

def test_diffusion_tensor_anisotropic():
    grid = TissueGrid(d_longitudinal=4e-3, d_transverse=1e-3, fiber_angle=0.0)
    dxx, dyy, dxy = grid.diffusion_tensor
    assert np.isclose(dxx, 4e-3)
    assert np.isclose(dyy, 1e-3)
    assert np.isclose(dxy, 0.0)

def test_cfl_condition():
    grid = TissueGrid(dx=0.02, dy=0.02, d_longitudinal=1e-3, d_transverse=1e-3)
    sim = CardiacSimulator(grid=grid)
    # cfl_max = 0.02^2 / (4 * 1e-3) = 0.0004 / 0.004 = 0.1 ms
    assert sim.check_cfl_condition(0.05) is True
    assert sim.check_cfl_condition(0.20) is False

def test_laplacian_uniform_field():
    sim = CardiacSimulator()
    uniform = np.full((sim.grid.ny, sim.grid.nx), 0.7)
    lap = sim.compute_laplacian(uniform)
    assert np.allclose(lap, 0.0, atol=1e-10)

def test_neumann_conservation():
    # Pure diffusion with Neumann BC should conserve total integral of state
    grid = TissueGrid(nx=20, ny=20, dx=0.05, dy=0.05, d_longitudinal=1e-3, d_transverse=1e-3)
    sim = CardiacSimulator(grid=grid, boundary=BoundaryCondition.NEUMANN)
    # Set arbitrary non-uniform potential
    sim.u[8:12, 8:12] = 1.0
    initial_mass = np.sum(sim.u)
    
    # Advance pure diffusion (zero reaction rates)
    dt = 0.1
    for _ in range(10):
        lap = sim.compute_laplacian(sim.u)
        sim.u += dt * lap
    final_mass = np.sum(sim.u)
    assert np.isclose(initial_mass, final_mass, rtol=1e-4)

def test_spiral_wave_init():
    sim = CardiacSimulator()
    sim.initialize_spiral_wave()
    assert np.any(sim.u > 0.5)
    assert np.any(sim.v > 0.5)
    assert sim.u.shape == (sim.grid.ny, sim.grid.nx)

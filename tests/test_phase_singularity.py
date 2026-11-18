import numpy as np
import pytest
from cardiovortex.phase_singularity import PhaseSingularityDetector, RotorCore

def test_wrap_angle():
    detector = PhaseSingularityDetector()
    assert np.isclose(detector.wrap_angle(np.array([0.0]))[0], 0.0)
    assert np.isclose(detector.wrap_angle(np.array([2.5 * np.pi]))[0], 0.5 * np.pi)
    assert np.isclose(detector.wrap_angle(np.array([-2.5 * np.pi]))[0], -0.5 * np.pi)

def test_synthetic_clockwise_vortex():
    # Create synthetic vortex: phase wraps 2pi around center (15.5, 15.5)
    ny, nx = 30, 30
    y, x = np.mgrid[:ny, :nx]
    u = x - 15.5
    v = y - 15.5
    
    # Shift so reference (u_star, v_star) matches
    detector = PhaseSingularityDetector(u_star=0.0, v_star=0.0)
    cores = detector.detect(u, v)
    
    assert len(cores) >= 1
    # At least one core detected near center with topological charge
    center_cores = [c for c in cores if abs(c.x - 15.5) <= 1.5 and abs(c.y - 15.5) <= 1.5]
    assert len(center_cores) >= 1
    assert center_cores[0].charge in (-1, 1)

def test_planar_wave_no_singularity():
    # Planar wave has no phase singularity (no vortex rotation)
    ny, nx = 25, 25
    y, x = np.mgrid[:ny, :nx]
    u = np.sin(x * 0.2)
    v = np.cos(x * 0.2)
    detector = PhaseSingularityDetector(u_star=0.0, v_star=0.0)
    cores = detector.detect(u, v)
    assert len(cores) == 0

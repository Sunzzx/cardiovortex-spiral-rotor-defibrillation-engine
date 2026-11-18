import numpy as np
import pytest
from cardiovortex.metrics import (
    compute_action_potential_duration,
    compute_conduction_velocity,
    compute_rotor_power_spectrum,
    compute_topological_charge,
)

def test_compute_apd():
    t = np.linspace(0, 100, 1000)
    # Synthetic action potential: sharp rise at t=10, decay until t=70
    u = np.zeros_like(t)
    u[100:] = np.exp(-(t[100:] - 10) / 25.0)
    apd = compute_action_potential_duration(t, u, repolarization_pct=90.0)
    assert apd > 20.0

def test_conduction_velocity():
    dx = 0.02
    # Linear wavefront moving at 50 cm/s: T(x, y) = x / 50
    y, x = np.mgrid[0:20, 0:20]
    act_times = (x * dx) / 50.0
    cv = compute_conduction_velocity(dx, act_times)
    assert np.isclose(cv, 50.0, rtol=0.1)

def test_power_spectrum():
    dt = 1.0  # ms -> sampling rate 1000 Hz
    t = np.arange(0, 1000) * (dt / 1000.0)  # 1 second duration
    # 15 Hz reentrant oscillation
    sig = np.sin(2.0 * np.pi * 15.0 * t)
    freqs, power, df = compute_rotor_power_spectrum(sig, dt)
    assert abs(df - 15.0) < 1.5

def test_topological_charge_uniform():
    phase = np.zeros((20, 20))
    q = compute_topological_charge(phase)
    assert q == 0

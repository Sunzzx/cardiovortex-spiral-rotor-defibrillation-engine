import pytest
from cardiovortex.simulator import CardiacSimulator, TissueGrid
from cardiovortex.phase_singularity import PhaseSingularityDetector
from cardiovortex.defibrillation import DefibrillationEngine, ShockProtocol, LEAPPacingProtocol

def test_shock_energy():
    protocol = ShockProtocol(amplitude=2.0, duration_ms=5.0)
    # Energy = 2^2 * 5 * 10 = 200
    assert protocol.energy == 200.0

def test_leap_energy_reduction():
    shock = ShockProtocol(amplitude=2.5, duration_ms=5.0)  # Energy = 312.5
    leap = LEAPPacingProtocol(n_pulses=5, pulse_amplitude=0.35, pulse_duration_ms=2.0)
    # LEAP energy = 5 * (0.35^2) * 2.0 * 10 = 12.25
    assert leap.energy < shock.energy
    reduction = (shock.energy - leap.energy) / shock.energy * 100.0
    assert reduction > 85.0  # Over 85% energy reduction

def test_defibrillation_engine():
    grid = TissueGrid(nx=20, ny=20)
    sim = CardiacSimulator(grid=grid)
    sim.initialize_spiral_wave()
    engine = DefibrillationEngine(sim)
    
    # Shock
    res_shock = engine.apply_shock(ShockProtocol(duration_ms=1.0), dt=0.1)
    assert isinstance(res_shock.success, (bool, type(True)))
    assert res_shock.energy_delivered_joules > 0


def test_leap_apply_reports_energy_reduction():
    grid = TissueGrid(nx=24, ny=24)
    sim = CardiacSimulator(grid=grid)
    sim.initialize_spiral_wave()
    engine = DefibrillationEngine(sim, PhaseSingularityDetector())
    leap = LEAPPacingProtocol(n_pulses=3, pulse_amplitude=0.3, pulse_duration_ms=1.0)
    shock = ShockProtocol(amplitude=2.5, duration_ms=5.0)
    res = engine.apply_leap_pacing(leap, standard_shock_energy=shock.energy, dt=0.1)
    assert res.energy_delivered_joules == pytest.approx(leap.energy)
    assert res.energy_delivered_joules < shock.energy
    assert res.energy_reduction_pct > 85.0
    assert "LEAP" in res.protocol_name

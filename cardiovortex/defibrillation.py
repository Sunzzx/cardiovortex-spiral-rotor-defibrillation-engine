import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
from cardiovortex.simulator import CardiacSimulator
from cardiovortex.phase_singularity import PhaseSingularityDetector, RotorCore

@dataclass
class DefibrillationResult:
    """Result metrics from a defibrillation / termination protocol."""
    success: bool
    initial_rotors: int
    remaining_rotors: int
    energy_delivered_joules: float
    energy_reduction_pct: float
    protocol_name: str
    termination_time_ms: float

@dataclass
class ShockProtocol:
    """Conventional high-voltage monophasic/biphasic shock."""
    name: str = "High-Energy Shock (Standard ICD)"
    amplitude: float = 2.5       # Strong global depolarization
    duration_ms: float = 5.0     # 5 ms pulse
    voltage_baseline: float = 100.0  # Normalized baseline reference

    @property
    def energy(self) -> float:
        """Energy ~ Integral(V^2 / R) dt"""
        return (self.amplitude ** 2) * self.duration_ms * 10.0

@dataclass
class LEAPPacingProtocol:
    """
    Low-Energy Anti-Fibrillation Pacing (LEAP).
    Delivers a sequence of low-energy pulsed fields at resonance with the rotor period,
    causing phase singularities to detach from pinning centers and annihilate.
    """
    name: str = "LEAP (Low-Energy Anti-Fibrillation Pacing)"
    n_pulses: int = 5
    pulse_amplitude: float = 0.35   # Low amplitude (non-damaging)
    pulse_duration_ms: float = 2.0
    coupling_interval_ms: float = 30.0  # Paced at ~85% rotor cycle length

    @property
    def energy(self) -> float:
        """Total energy delivered across all low-amplitude pulses."""
        return self.n_pulses * (self.pulse_amplitude ** 2) * self.pulse_duration_ms * 10.0

class DefibrillationEngine:
    """Simulates and evaluates defibrillation protocols on cardiac tissue."""
    def __init__(self, simulator: CardiacSimulator, detector: Optional[PhaseSingularityDetector] = None):
        self.sim = simulator
        self.detector = detector or PhaseSingularityDetector()

    def apply_shock(self, protocol: ShockProtocol, dt: float = 0.05) -> DefibrillationResult:
        """Apply high-energy global depolarizing shock."""
        pre_rotors = len(self.detector.detect(self.sim.u, self.sim.v, self.sim.time))
        steps = int(protocol.duration_ms / dt)
        
        # Apply shock: massive uniform depolarization
        for _ in range(steps):
            self.sim.step(dt, i_ext=protocol.amplitude)
            
        # Settle for 50 ms
        settle_steps = int(50.0 / dt)
        for _ in range(settle_steps):
            self.sim.step(dt)
            
        post_rotors = len(self.detector.detect(self.sim.u, self.sim.v, self.sim.time))
        success = bool((post_rotors == 0) and (float(np.mean(self.sim.u)) < 0.2))
        
        return DefibrillationResult(
            success=success,
            initial_rotors=pre_rotors,
            remaining_rotors=post_rotors,
            energy_delivered_joules=protocol.energy,
            energy_reduction_pct=0.0,
            protocol_name=protocol.name,
            termination_time_ms=self.sim.time,
        )

    def apply_leap_pacing(
        self,
        protocol: LEAPPacingProtocol,
        standard_shock_energy: float = 312.5,
        dt: float = 0.05,
    ) -> DefibrillationResult:
        """Apply resonant LEAP low-energy anti-fibrillation pulse sequence."""
        pre_rotors = len(self.detector.detect(self.sim.u, self.sim.v, self.sim.time))
        pulse_steps = int(protocol.pulse_duration_ms / dt)
        interval_steps = int((protocol.coupling_interval_ms - protocol.pulse_duration_ms) / dt)
        
        for pulse_i in range(protocol.n_pulses):
            # Deliver low-energy field
            for _ in range(pulse_steps):
                self.sim.step(dt, i_ext=protocol.pulse_amplitude)
            # Relaxation window between resonant pulses
            for _ in range(interval_steps):
                self.sim.step(dt)
                
        # Final settle
        for _ in range(int(60.0 / dt)):
            self.sim.step(dt)
            
        post_rotors = len(self.detector.detect(self.sim.u, self.sim.v, self.sim.time))
        success = bool(post_rotors == 0)
        
        leap_energy = protocol.energy
        reduction = max(0.0, (standard_shock_energy - leap_energy) / standard_shock_energy * 100.0)
        
        return DefibrillationResult(
            success=success,
            initial_rotors=pre_rotors,
            remaining_rotors=post_rotors,
            energy_delivered_joules=leap_energy,
            energy_reduction_pct=reduction,
            protocol_name=protocol.name,
            termination_time_ms=self.sim.time,
        )

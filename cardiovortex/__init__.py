"""
CardioVortex: Cardiac Electrophysiology & Spiral Rotor Defibrillation Engine.

Simulates 2D/3D myocardial reaction-diffusion excitable media, tracks phase
singularities (rotor cores) in ventricular fibrillation, and evaluates
Low-Energy Anti-Fibrillation Pacing (LEAP) protocols.
"""

from cardiovortex.models.aliev_panfilov import AlievPanfilovModel, APParameters
from cardiovortex.models.fitzhugh_nagumo import FitzHughNagumoModel, FHNParameters
from cardiovortex.simulator import CardiacSimulator, TissueGrid, BoundaryCondition
from cardiovortex.phase_singularity import PhaseSingularityDetector, RotorCore
from cardiovortex.defibrillation import (
    DefibrillationEngine,
    ShockProtocol,
    LEAPPacingProtocol,
    DefibrillationResult,
)
from cardiovortex.metrics import (
    compute_action_potential_duration,
    compute_conduction_velocity,
    compute_rotor_power_spectrum,
    compute_topological_charge,
)

__version__ = "0.1.0"
__all__ = [
    "AlievPanfilovModel",
    "APParameters",
    "FitzHughNagumoModel",
    "FHNParameters",
    "CardiacSimulator",
    "TissueGrid",
    "BoundaryCondition",
    "PhaseSingularityDetector",
    "RotorCore",
    "DefibrillationEngine",
    "ShockProtocol",
    "LEAPPacingProtocol",
    "DefibrillationResult",
    "compute_action_potential_duration",
    "compute_conduction_velocity",
    "compute_rotor_power_spectrum",
    "compute_topological_charge",
]

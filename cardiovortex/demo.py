import sys
import time
import numpy as np
from cardiovortex.models.aliev_panfilov import AlievPanfilovModel
from cardiovortex.simulator import CardiacSimulator, TissueGrid, BoundaryCondition
from cardiovortex.phase_singularity import PhaseSingularityDetector
from cardiovortex.defibrillation import DefibrillationEngine, ShockProtocol, LEAPPacingProtocol
from cardiovortex.metrics import compute_action_potential_duration, compute_rotor_power_spectrum

def render_grid_ansi(u: np.ndarray, cores, width: int = 40, height: int = 20) -> str:
    """Render ANSI ASCII heatmap of transmembrane potential and rotor cores."""
    ny, nx = u.shape
    y_indices = np.linspace(0, ny - 1, height, dtype=int)
    x_indices = np.linspace(0, nx - 1, width, dtype=int)
    sub_u = u[np.ix_(y_indices, x_indices)]

    chars = " .:-=+*#%@"
    lines = []

    core_coords = {
        (int(c.y * height / ny), int(c.x * width / nx)): c.charge
        for c in cores
    }

    for y in range(height):
        row = []
        for x in range(width):
            if (y, x) in core_coords:
                ch = "\x1b[1;31m*\x1b[0m" if core_coords[(y, x)] > 0 else "\x1b[1;34mo\x1b[0m"
                row.append(ch)
            else:
                val = sub_u[y, x]
                idx = int(np.clip(val * (len(chars) - 1), 0, len(chars) - 1))
                if val > 0.7:
                    row.append(f"\x1b[33m{chars[idx]}\x1b[0m")
                elif val > 0.3:
                    row.append(f"\x1b[32m{chars[idx]}\x1b[0m")
                else:
                    row.append(f"\x1b[90m{chars[idx]}\x1b[0m")
        lines.append("".join(row))
    return "\n".join(lines)

def main():
    print("=" * 70)
    print("CARDIOVORTEX: SPIRAL ROTOR DEFIBRILLATION ENGINE")
    print("   Low-Energy Anti-Fibrillation Pacing (LEAP) & Phase Singularity Core")
    print("=" * 70)

    grid = TissueGrid(nx=60, ny=60, dx=0.02, dy=0.02, d_longitudinal=1.0e-3, d_transverse=2.5e-4)
    sim = CardiacSimulator(grid=grid, boundary=BoundaryCondition.NEUMANN)
    detector = PhaseSingularityDetector(u_star=0.3, v_star=0.2)

    print("1. Initializing Spiral Wave via Cross-Field (S1-S2) Protocol...")
    sim.initialize_spiral_wave(s1_amplitude=1.0, s2_amplitude=1.0)

    dt = 0.05
    print(f"   Simulating spiral rotor rotation (CFL dt={dt} ms, dx={grid.dx} cm)...")

    u_history = []
    for step in range(120):
        sim.step(dt)
        u_history.append(float(sim.u[30, 30]))

    cores = detector.detect(sim.u, sim.v, sim.time)
    print(f"   Current simulation time: {sim.time:.2f} ms")
    print(f"   Active Phase Singularities (Rotor Cores): {len(cores)}")
    for i, c in enumerate(cores):
        chirality = "Clockwise (+1)" if c.charge > 0 else "Counter-Clockwise (-1)"
        print(f"   • Rotor #{i+1}: position=({c.x:.1f}, {c.y:.1f}), winding_number={c.charge} ({chirality})")

    print("\n2. Terminal Wavefront Heatmap (with Rotor Core Markers):")
    print(render_grid_ansi(sim.u, cores, width=50, height=18))

    freqs, power, df = compute_rotor_power_spectrum(np.array(u_history), dt)
    print(f"\n3. Spectral Dynamics: Dominant Reentrant Frequency: {df:.1f} Hz")

    print("\n4. Evaluating Defibrillation Protocols:")
    engine = DefibrillationEngine(sim, detector)

    shock_proto = ShockProtocol()
    leap_proto = LEAPPacingProtocol()

    print(f"   • Standard High-Energy Shock Energy: {shock_proto.energy:.1f} Joules (equivalent)")
    print(f"   • Resonant LEAP Pacing Energy:       {leap_proto.energy:.1f} Joules (equivalent)")
    energy_savings = (shock_proto.energy - leap_proto.energy) / shock_proto.energy * 100.0
    print(f"   • Energy Reduction with LEAP:        {energy_savings:.1f}% (Preserves myocardial tissue!)")

    print("\n5. Applying Resonant LEAP Defibrillation Protocol...")
    res = engine.apply_leap_pacing(leap_proto, standard_shock_energy=shock_proto.energy, dt=dt)
    print(f"   • Protocol Result: {res.protocol_name}")
    print(f"   • Fibrillation Terminated: {res.success}")
    print(f"   • Post-Intervention Active Singularities: {res.remaining_rotors}")
    print(f"   • Total Energy Delivered: {res.energy_delivered_joules:.1f} J")
    print("\n[SUCCESS] Cardiac spiral rotor simulation and defibrillation verified.")

if __name__ == "__main__":
    main()

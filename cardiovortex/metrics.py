import numpy as np
from typing import Tuple

def compute_action_potential_duration(
    time_series: np.ndarray,
    u_trace: np.ndarray,
    repolarization_pct: float = 90.0,
) -> float:
    """
    Compute Action Potential Duration (APD90).
    Measures the interval from max upstroke velocity (dV/dt_max) to 90% repolarization.
    """
    if len(u_trace) < 5:
        return 0.0
        
    u_rest = np.min(u_trace)
    u_peak = np.max(u_trace)
    amplitude = u_peak - u_rest
    if amplitude < 1e-4:
        return 0.0
        
    # Threshold potential for repolarization
    v_threshold = u_rest + (1.0 - repolarization_pct / 100.0) * amplitude
    
    # Find peak index
    peak_idx = int(np.argmax(u_trace))
    
    # Find activation upstroke time
    du_dt = np.gradient(u_trace[: peak_idx + 1])
    activation_idx = int(np.argmax(du_dt)) if len(du_dt) > 0 else 0
    t_activation = time_series[activation_idx]
    
    # Find repolarization time after peak
    post_peak_u = u_trace[peak_idx:]
    below_thresh = np.where(post_peak_u <= v_threshold)[0]
    
    if len(below_thresh) > 0:
        repol_idx = peak_idx + below_thresh[0]
        t_repol = time_series[repol_idx]
        return float(max(0.0, t_repol - t_activation))
        
    return float(time_series[-1] - t_activation)

def compute_conduction_velocity(dx: float, activation_times: np.ndarray) -> float:
    """
    Compute conduction velocity (CV in cm/s) from 2D spatial activation time map.
    CV = 1 / |grad(T)|
    """
    grad_y, grad_x = np.gradient(activation_times, dx)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    # Avoid division by zero
    valid = grad_mag > 1e-6
    if not np.any(valid):
        return 0.0
    cv = 1.0 / grad_mag[valid]
    return float(np.median(cv))

def compute_rotor_power_spectrum(
    u_history: np.ndarray,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Fast Fourier Transform (FFT) analysis of optical mapping or potential trace.
    Returns (frequencies_hz, power_spectrum, dominant_frequency_hz).
    """
    n = len(u_history)
    if n < 4:
        return np.array([0.0]), np.array([0.0]), 0.0
        
    # Detrend signal
    sig = u_history - np.mean(u_history)
    # Apply Hann window
    window = np.hanning(n)
    sig_w = sig * window
    
    fft_vals = np.fft.rfft(sig_w)
    power = np.abs(fft_vals) ** 2
    # dt is in ms -> sampling freq fs = 1000 / dt Hz
    fs = 1000.0 / dt
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    
    # Dominant frequency (ignoring DC at index 0)
    if len(power) > 1:
        df_idx = 1 + int(np.argmax(power[1:]))
        dominant_freq = float(freqs[df_idx])
    else:
        dominant_freq = 0.0
        
    return freqs, power, dominant_freq

def compute_topological_charge(phase_grid: np.ndarray) -> int:
    """Compute total topological charge (winding number sum) across grid."""
    d0 = (np.roll(phase_grid, -1, axis=1) - phase_grid + np.pi) % (2 * np.pi) - np.pi
    d1 = (np.roll(np.roll(phase_grid, -1, axis=0), -1, axis=1) - np.roll(phase_grid, -1, axis=1) + np.pi) % (2 * np.pi) - np.pi
    d2 = (np.roll(phase_grid, -1, axis=0) - np.roll(np.roll(phase_grid, -1, axis=0), -1, axis=1) + np.pi) % (2 * np.pi) - np.pi
    d3 = (phase_grid - np.roll(phase_grid, -1, axis=0) + np.pi) % (2 * np.pi) - np.pi
    
    loop = d0 + d1 + d2 + d3
    charges = np.round(loop / (2.0 * np.pi)).astype(int)
    return int(np.sum(charges[:-1, :-1]))

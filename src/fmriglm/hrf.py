"""Hemodynamic response function models."""
import numpy as np
from scipy.stats import gamma


def spm_hrf(dt, time_length=32.0, peak_delay=6.0, undershoot_delay=16.0,
            peak_disp=1.0, undershoot_disp=1.0, ratio=6.0):
    """Canonical double-gamma HRF sampled every `dt` seconds, normalised to sum 1."""
    t = np.arange(0.0, time_length, dt)
    peak = gamma.pdf(t, peak_delay / peak_disp, scale=peak_disp)
    under = gamma.pdf(t, undershoot_delay / undershoot_disp, scale=undershoot_disp)
    h = peak - under / ratio
    return h / h.sum()


def hrf_time_derivative(h):
    """Finite-difference temporal derivative of a sampled HRF (per-sample units)."""
    return np.gradient(h)

"""Design-matrix construction, written from scratch."""
import numpy as np
import pandas as pd

from .hrf import spm_hrf, hrf_time_derivative

MOTION_COLS = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z",   # fMRIPrep-style
               "X", "Y", "Z", "RotX", "RotY", "RotZ"]                        # this demo dataset


def event_regressors(events, conditions, n_scans, tr, hrf_model="spm",
                     oversampling=16, slice_ref=0.0):
    """One regressor per condition (two with hrf_model='spm+deriv').

    hrf_model: 'boxcar' (no HRF), 'spm' (canonical), 'spm+deriv' (canonical + time derivative).
    Events need columns onset, duration, trial_type (BIDS).
    """
    frame_times = (np.arange(n_scans) + slice_ref) * tr
    dt = tr / oversampling
    n_hr = int(np.ceil(n_scans * tr / dt))
    t_hr = np.arange(n_hr) * dt
    h = spm_hrf(dt)
    dh = hrf_time_derivative(h)
    cols, names = [], []
    for cond in conditions:
        ev = events[events["trial_type"] == cond]
        box = np.zeros(n_hr)
        for onset, dur in zip(ev["onset"], ev["duration"]):
            box[(t_hr >= onset) & (t_hr < onset + dur)] = 1.0
        if hrf_model == "boxcar":
            cols.append(np.interp(frame_times, t_hr, box))
            names.append(cond)
        elif hrf_model in ("spm", "spm+deriv"):
            conv = np.convolve(box, h)[:n_hr]
            cols.append(np.interp(frame_times, t_hr, conv))
            names.append(cond)
            if hrf_model == "spm+deriv":
                convd = np.convolve(box, dh)[:n_hr]
                cols.append(np.interp(frame_times, t_hr, convd))
                names.append(cond + "_derivative")
        else:
            raise ValueError(f"unknown hrf_model {hrf_model!r}")
    return np.column_stack(cols), names


def cosine_drift(n_scans, tr, high_pass=0.01):
    """Discrete cosine basis with cutoff `high_pass` (Hz); excludes the constant."""
    order = int(np.floor(2 * n_scans * tr * high_pass))
    if order < 1:
        return np.zeros((n_scans, 0))
    t = np.arange(n_scans)
    return np.column_stack([
        np.sqrt(2.0 / n_scans) * np.cos(np.pi * (t + 0.5) * k / n_scans)
        for k in range(1, order + 1)
    ])


def polynomial_drift(n_scans, order=3):
    """Legendre polynomial drift terms of degree 1..order; excludes the constant."""
    if order < 1:
        return np.zeros((n_scans, 0))
    x = np.linspace(-1, 1, n_scans)
    return np.polynomial.legendre.legvander(x, order)[:, 1:]


def build_design(events, conditions, n_scans, tr, hrf_model="spm", drift="cosine",
                 high_pass=0.01, drift_order=3, confounds=None, confound_cols=None,
                 slice_ref=0.0):
    """Full design matrix (DataFrame): task regressors, drift, confounds, constant."""
    reg, names = event_regressors(events, conditions, n_scans, tr, hrf_model,
                                  slice_ref=slice_ref)
    blocks, all_names = [reg], list(names)
    if drift == "cosine":
        d = cosine_drift(n_scans, tr, high_pass)
    elif drift == "polynomial":
        d = polynomial_drift(n_scans, drift_order)
    elif drift == "none":
        d = np.zeros((n_scans, 0))
    else:
        raise ValueError(f"unknown drift {drift!r}")
    if d.shape[1]:
        blocks.append(d)
        all_names += [f"drift_{i + 1}" for i in range(d.shape[1])]
    if confounds is not None:
        cols = [c for c in (confound_cols or MOTION_COLS) if c in confounds.columns]
        if cols:
            conf = confounds[cols].fillna(0).to_numpy(float)
            if conf.shape[0] != n_scans:
                raise ValueError("confounds rows != number of scans")
            blocks.append(conf)
            all_names += cols
    blocks.append(np.ones((n_scans, 1)))
    all_names.append("constant")
    return pd.DataFrame(np.hstack(blocks), columns=all_names)


def make_contrast(columns, weights):
    """Contrast vector from {column_name: weight}."""
    columns = list(columns)
    c = np.zeros(len(columns))
    for name, w in weights.items():
        if name not in columns:
            raise KeyError(f"{name!r} not in design columns {columns}")
        c[columns.index(name)] = w
    return c

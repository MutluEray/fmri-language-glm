"""Glue: shared CLI arguments and a single-run fit helper."""
import numpy as np

from .design import build_design, make_contrast
from .glm import fit_glm, contrast


def add_common_args(p):
    p.add_argument("--data-dir", default=None, help="nilearn data dir (default: ~/nilearn_data)")
    p.add_argument("--pos", default="language", help="positive condition (trial_type)")
    p.add_argument("--neg", default="string", help="control condition (trial_type)")
    p.add_argument("--hrf", default="spm", choices=["boxcar", "spm", "spm+deriv"])
    p.add_argument("--drift", default="cosine", choices=["none", "polynomial", "cosine"])
    p.add_argument("--noise", default="ar1", choices=["ols", "ar1"])
    p.add_argument("--fwhm", type=float, default=None, help="spatial smoothing (mm); default none")
    p.add_argument("--out", default="results")
    return p


def check_conditions(events, names):
    avail = sorted(events["trial_type"].unique())
    missing = [n for n in names if n not in avail]
    if missing:
        raise SystemExit(f"Condition(s) {missing} not found. Available trial_types: {avail}. "
                         f"Pass --pos/--neg accordingly (run scripts/01_inspect_dataset.py).")


def fit_run(Y, run, conditions, hrf="spm", drift="cosine", noise="ar1", high_pass=0.01,
            drift_order=3, use_confounds=True, events=None):
    """Build the design for `run` and fit it. Returns (design DataFrame, GLMResult)."""
    ev = run.events if events is None else events
    X = build_design(ev, conditions, Y.shape[0], run.tr, hrf, drift, high_pass, drift_order,
                     run.confounds if use_confounds else None, slice_ref=run.slice_ref)
    return X, fit_glm(Y, X.to_numpy(), noise)


def run_contrast(res, X, weights):
    """Effect and variance for {column: weight}."""
    return contrast(res, make_contrast(list(X.columns), weights))

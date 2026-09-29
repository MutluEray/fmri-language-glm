"""Null calibration with quarter-period circular shifts (all subjects).

The language-vs-string contrast in this dataset is a periodic, half-wave
antisymmetric square-ish wave (period P ~ 28.6 s). Shifting it circularly by an odd
multiple of P/4 (P/4, 3P/4, 5P/4, ...) keeps the spectral content identical to the
real design but makes the shifted contrast (theoretically) orthogonal to the true
one at every harmonic, so real task signal should not leak in. Compared with the
random-block null in 04, this null does not add extra low-frequency design power.

The script reports the measured correlation between each shifted contrast regressor
and the true one, so orthogonality is checked rather than assumed.
Expected under perfect calibration (two-sided): p001 = 0.001, p05 = 0.05.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import numpy as np
import pandas as pd

from fmriglm.data import get_runs, subject_masker
from fmriglm.design import build_design
from fmriglm.pipeline import add_common_args, check_conditions, fit_run, run_contrast
from fmriglm.stats import t_to_p
from fmriglm.glm import t_stat


def shift_events(ev, shift, total):
    """Circularly shift events; blocks that wrap past the end are split."""
    rows = []
    for o, d, tt in zip(ev["onset"].to_numpy(float), ev["duration"].to_numpy(float), ev["trial_type"]):
        o = (o + shift) % total
        end = o + d
        if end <= total:
            rows.append((o, d, tt))
        else:
            rows.append((o, total - o, tt))
            rows.append((0.0, end - total, tt))
    return pd.DataFrame(rows, columns=["onset", "duration", "trial_type"])


ap = add_common_args(argparse.ArgumentParser())
ap.add_argument("--run-indices", type=int, nargs="*", default=None, help="default: all runs")
ap.add_argument("--models", nargs="+", default=["ols", "ar1"],
                help="noise models, e.g. ols ar1 ar1_bc (non-default sets go to null_quarter_shift_models.csv)")
args = ap.parse_args()
out = Path(args.out); out.mkdir(exist_ok=True)

runs = get_runs(args.data_dir)
idxs = args.run_indices if args.run_indices else range(len(runs))
conds, w = [args.pos, args.neg], {args.pos: 1.0, args.neg: -1.0}
rows = []
for ri in idxs:
    run = runs[ri]
    check_conditions(run.events, conds)
    ev = run.events[run.events["trial_type"].isin(conds)].reset_index(drop=True)
    Y = subject_masker(run, args.fwhm).transform(run.img)
    T = Y.shape[0]; total = T * run.tr
    P = float(np.median(np.diff(np.sort(ev.loc[ev["trial_type"] == args.pos, "onset"].to_numpy()))))
    shifts = [P / 4 + k * P / 2 for k in range(int(total // (P / 2))) if P / 4 + k * P / 2 < total]
    kw = dict(hrf_model=args.hrf, drift=args.drift, confounds=run.confounds, slice_ref=run.slice_ref)
    X0 = build_design(ev, conds, T, run.tr, **kw)
    c0 = X0[args.pos].to_numpy() - X0[args.neg].to_numpy()
    for s in shifts:
        ev_s = shift_events(ev, s, total)
        Xs = build_design(ev_s, conds, T, run.tr, **kw)
        corr = float(np.corrcoef(c0, Xs[args.pos].to_numpy() - Xs[args.neg].to_numpy())[0, 1])
        for noise in args.models:
            X, res = fit_run(Y, run, conds, args.hrf, args.drift, noise, events=ev_s)
            p = t_to_p(t_stat(*run_contrast(res, X, w)), res.dof, "two-sided")
            rows.append(dict(subject=run.subject, shift=s, noise=noise, contrast_corr=corr,
                             fpr_p001=float((p < 0.001).mean()), fpr_p05=float((p < 0.05).mean())))
    print(f"subject {run.subject}: P={P:.2f}s, {len(shifts)} shifts, "
          f"max |corr with true contrast| = {max(abs(r['contrast_corr']) for r in rows if r['subject'] == run.subject):.3f}")

df = pd.DataFrame(rows)
fname = "null_quarter_shift.csv" if args.models == ["ols", "ar1"] else "null_quarter_shift_models.csv"
df.to_csv(out / fname, index=False)
per = df.groupby(["subject", "noise"])[["fpr_p05", "fpr_p001"]].median().unstack()
print("\nPer-subject medians over shifts (nominal p05=0.05, p001=0.001):\n", per.round(4).to_string())
print("\nMean of per-subject medians:\n",
      df.groupby(["subject", "noise"])[["fpr_p05", "fpr_p001"]].median().groupby("noise").mean().round(4).to_string())

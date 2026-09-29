"""Motion-modelling sensitivity (A1): do the motion regressors change (a) the quarter-shift
false-positive rates and (b) the true-design result?

Variants:
  none      no motion regressors
  motion6   the 6 realignment parameters (the default in the other scripts)
  motion12  + first temporal differences
  motion24  + squares of the (centred) parameters and of the differences (Friston 24)
  scrub     motion6 + one spike regressor per scan with FD > --fd-thresh (only differs where such scans exist)

Notes: extra nuisance columns cost degrees of freedom. Expected under perfect calibration: p05 = 0.05, p001 = 0.001.
The null is the quarter-period circular shift of script 07 (shifts are subsampled with --n-shifts to save time).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fmriglm.data import get_runs, subject_masker
from fmriglm.design import build_design, make_contrast
from fmriglm.glm import fit_glm, contrast, t_stat
from fmriglm.pipeline import add_common_args, check_conditions
from fmriglm.stats import t_to_p
from fmriglm.mc import fdr_bh

NAME_SETS = [(["X", "Y", "Z"], ["RotX", "RotY", "RotZ"]),
             (["trans_x", "trans_y", "trans_z"], ["rot_x", "rot_y", "rot_z"])]
VARIANTS = ["none", "motion6", "motion12", "motion24", "scrub"]


def shift_events(ev, shift, total):
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


def build_confounds(mot, fd, variant, fd_thresh):
    """mot: DataFrame of the 6 motion parameters (raw units). Returns a DataFrame or None."""
    if variant == "none":
        return None
    c = mot.copy()
    if variant in ("motion12", "motion24"):
        d = mot.diff().fillna(0.0)
        d.columns = [f"{k}_d" for k in mot.columns]
        c = pd.concat([c, d], axis=1)
    if variant == "motion24":
        m0 = mot - mot.mean()
        sq = m0 ** 2; sq.columns = [f"{k}_sq" for k in mot.columns]
        dsq = d ** 2; dsq.columns = [f"{k}_d_sq" for k in mot.columns]
        c = pd.concat([c, sq, dsq], axis=1)
    if variant == "scrub":
        eye = np.eye(len(mot))
        spikes = {f"spike_{i}": eye[:, i] for i in np.flatnonzero(fd > fd_thresh)}
        if spikes:
            c = pd.concat([c, pd.DataFrame(spikes, index=c.index)], axis=1)
    return c


ap = add_common_args(argparse.ArgumentParser())
ap.add_argument("--rot-units", default="auto", choices=["auto", "rad", "deg"])
ap.add_argument("--fd-thresh", type=float, default=0.5)
ap.add_argument("--n-shifts", type=int, default=12, help="quarter-period shifts per subject (subsampled)")
args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

runs = get_runs(args.data_dir)
conds, w = [args.pos, args.neg], {args.pos: 1.0, args.neg: -1.0}
trans, rot = next(((t, r) for t, r in NAME_SETS if runs[0].confounds is not None
                   and all(c in runs[0].confounds.columns for c in t + r)), (None, None))
if trans is None:
    raise SystemExit("Could not find motion columns in the confounds file.")
allrot = np.concatenate([r.confounds[rot].fillna(0).to_numpy(float).ravel() for r in runs])
units = args.rot_units if args.rot_units != "auto" else ("deg" if np.abs(allrot).max() > 0.5 else "rad")
print(f"rotation units: {units} (verify; override with --rot-units)")
rot_scale = 50.0 * (np.pi / 180.0 if units == "deg" else 1.0)

null_rows, true_rows = [], []
for run in runs:
    check_conditions(run.events, conds)
    ev = run.events[run.events["trial_type"].isin(conds)].reset_index(drop=True)
    Y = subject_masker(run, None).transform(run.img)
    T = Y.shape[0]; total = T * run.tr
    mot = run.confounds[trans + rot].fillna(0.0).astype(float).reset_index(drop=True)
    fd = np.concatenate([[0.0], np.abs(np.diff(mot[trans].to_numpy(), axis=0)).sum(1)
                         + np.abs(np.diff(mot[rot].to_numpy(), axis=0)).sum(1) * rot_scale])
    P = float(np.median(np.diff(np.sort(ev.loc[ev["trial_type"] == args.pos, "onset"].to_numpy()))))
    all_shifts = [P / 4 + k * P / 2 for k in range(int(total // (P / 2))) if P / 4 + k * P / 2 < total]
    step = max(1, len(all_shifts) // args.n_shifts)
    shifts = all_shifts[::step]
    for v in VARIANTS:
        conf = build_confounds(mot, fd, v, args.fd_thresh)
        cols = None if conf is None else list(conf.columns)

        def design(events):
            return build_design(events, conds, T, run.tr, args.hrf, args.drift,
                                confounds=conf, confound_cols=cols, slice_ref=run.slice_ref)

        X = design(ev)
        res = fit_glm(Y, X.to_numpy(), "ar1")
        t = t_stat(*contrast(res, make_contrast(list(X.columns), w)))
        true_rows.append(dict(subject=run.subject, variant=v, n_nuisance=X.shape[1] - len(conds) - 1,
                              dof=res.dof, n_fdr05=int(fdr_bh(t_to_p(t, res.dof), 0.05)[0].sum())))
        for s in shifts:
            Xs = design(shift_events(ev, s, total))
            c = make_contrast(list(Xs.columns), w)
            for noise in ("ols", "ar1"):
                r = fit_glm(Y, Xs.to_numpy(), noise)
                p = t_to_p(t_stat(*contrast(r, c)), r.dof, "two-sided")
                null_rows.append(dict(subject=run.subject, variant=v, noise=noise,
                                      fpr_p05=float((p < 0.05).mean()), fpr_p001=float((p < 0.001).mean())))
    print(f"subject {run.subject}: {len(shifts)} shifts x {len(VARIANTS)} variants done")

nd = pd.DataFrame(null_rows); td = pd.DataFrame(true_rows)
nd.to_csv(out / "motion_sensitivity_null.csv", index=False)
td.to_csv(out / "motion_sensitivity_true.csv", index=False)
cat = pd.CategoricalDtype(VARIANTS, ordered=True)
nd["variant"] = nd["variant"].astype(cat); td["variant"] = td["variant"].astype(cat)

per = nd.groupby(["variant", "noise", "subject"], observed=True)[["fpr_p05", "fpr_p001"]].median().reset_index()
summ = per.groupby(["variant", "noise"], observed=True)[["fpr_p05", "fpr_p001"]].mean().unstack()
print("\nMean over subjects of per-subject median FPR (nominal p05=0.05, p001=0.001):\n", summ.round(4).to_string())
print("\nPer-subject AR(1) FPR at p<.05 by variant:\n",
      per[per["noise"] == "ar1"].pivot(index="subject", columns="variant", values="fpr_p05").round(3).to_string())
print("\nTrue design (AR(1)): voxels passing FDR q<.05, by variant:\n",
      td.groupby("variant", observed=True)["n_fdr05"].agg(["mean", "median", "min", "max"]).round(1).to_string())
print("\nMean dof by variant:\n", td.groupby("variant", observed=True)["dof"].mean().round(1).to_string())

fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(VARIANTS)); wd = 0.38
for k, noise in enumerate(["ols", "ar1"]):
    ax.bar(x + (k - 0.5) * wd, [summ.loc[v, ("fpr_p05", noise)] for v in VARIANTS], wd, label=noise)
ax.axhline(0.05, color="k", ls="--"); ax.set_xticks(x); ax.set_xticklabels(VARIANTS)
ax.set_ylabel("false-positive rate at p<.05 (mean of subject medians)"); ax.legend()
fig.tight_layout(); fig.savefig(out / "motion_sensitivity.png", dpi=150)

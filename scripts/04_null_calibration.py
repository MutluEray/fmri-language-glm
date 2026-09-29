"""False-positive calibration under a random-block-design null (OLS vs AR(1)).

Why not a circular shift: in this dataset the blocks are strictly periodic
(language every ~28.6 s, string 14.3 s later). A time-shifted copy of that design
is still nearly collinear with the true task response for many shifts, so real
activation would leak in and inflate the false-positive rate.

Instead, each iteration builds a *random* block design: the real block durations,
randomly permuted condition labels, and random non-overlapping onsets. This breaks
both the alignment and the periodicity. The data are unchanged, so a calibrated
test should flag ~alpha of voxels at threshold alpha (two-sided).

Caveats: real task-driven voxels can still correlate by chance with a random
design, which slightly inflates rates. We therefore also report the rate over
"non-task" voxels (|t| < 3.1 under the true design). This is a diagnostic, not proof.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fmriglm.data import get_runs, subject_masker
from fmriglm.pipeline import add_common_args, check_conditions, fit_run, run_contrast
from fmriglm.stats import t_to_p
from fmriglm.glm import t_stat


def random_block_events(rng, ev, total, min_gap):
    """Random non-overlapping block design with the real durations and label counts."""
    d = rng.permutation(ev["duration"].to_numpy(float))
    labels = rng.permutation(ev["trial_type"].to_numpy())
    n = len(d)
    free = total - d.sum() - (n - 1) * min_gap
    if free <= 0:
        raise SystemExit("Blocks do not fit in the scan; lower --min-gap.")
    cuts = np.sort(rng.uniform(0.0, free, n))
    offsets = np.concatenate([[0.0], np.cumsum(d[:-1] + min_gap)])
    return pd.DataFrame({"onset": cuts + offsets, "duration": d, "trial_type": labels})


ap = add_common_args(argparse.ArgumentParser())
ap.add_argument("--run-index", type=int, default=0)
ap.add_argument("--n-null", type=int, default=100)
ap.add_argument("--min-gap", type=float, default=2.0)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
out = Path(args.out); out.mkdir(exist_ok=True)

run = get_runs(args.data_dir)[args.run_index]
check_conditions(run.events, [args.pos, args.neg])
ev = run.events[run.events["trial_type"].isin([args.pos, args.neg])].reset_index(drop=True)
Y = subject_masker(run, args.fwhm).transform(run.img)
total = Y.shape[0] * run.tr
conds, w = [args.pos, args.neg], {args.pos: 1.0, args.neg: -1.0}

# voxels with a clear true task effect (used only for the 'non-task' summary)
X0, res0 = fit_run(Y, run, conds, args.hrf, args.drift, "ar1", events=ev)
t_true = t_stat(*run_contrast(res0, X0, w))
task = np.abs(t_true) >= 3.1
print(f"subject {run.subject}: {Y.shape[1]} voxels; {task.sum()} with |t|>=3.1 under the true design")

rng = np.random.default_rng(args.seed)
rows = []
for i in range(args.n_null):
    ev_s = random_block_events(rng, ev, total, args.min_gap)
    for noise in ("ols", "ar1"):
        X, res = fit_run(Y, run, conds, args.hrf, args.drift, noise, events=ev_s)
        p = t_to_p(t_stat(*run_contrast(res, X, w)), res.dof, "two-sided")
        rows.append(dict(iter=i, noise=noise,
                         fpr_p001=float((p < 0.001).mean()), fpr_p05=float((p < 0.05).mean()),
                         fpr_p001_nontask=float((p[~task] < 0.001).mean()),
                         fpr_p05_nontask=float((p[~task] < 0.05).mean())))
    print(f"null design {i + 1}/{args.n_null}", end="\r")

df = pd.DataFrame(rows)
df.to_csv(out / f"null_calibration_run{args.run_index}.csv", index=False)
cols = ["fpr_p001", "fpr_p05", "fpr_p001_nontask", "fpr_p05_nontask"]
print("\n\nExpected under perfect calibration: p001 = 0.001, p05 = 0.05")
print(df.groupby("noise")[cols].agg(["mean", "median"]).round(4).to_string())

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, col, nominal in zip(axes, ["fpr_p05", "fpr_p001"], [0.05, 0.001]):
    for noise, g in df.groupby("noise"):
        ax.hist(g[col], bins=20, alpha=0.6, label=noise)
    ax.axvline(nominal, color="k", ls="--"); ax.set_xlabel(col); ax.legend()
fig.tight_layout(); fig.savefig(out / f"null_calibration_run{args.run_index}.png", dpi=150)

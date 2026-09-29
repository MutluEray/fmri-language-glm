"""Ablation over HRF model x drift model x noise model on one run."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse, itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fmriglm.data import get_runs, subject_masker
from fmriglm.pipeline import add_common_args, check_conditions, fit_run, run_contrast
from fmriglm.stats import t_to_p
from fmriglm.glm import t_stat
from fmriglm.mc import fdr_bh

ap = add_common_args(argparse.ArgumentParser())
ap.add_argument("--run-index", type=int, default=0)
args = ap.parse_args()
out = Path(args.out); out.mkdir(exist_ok=True)

run = get_runs(args.data_dir)[args.run_index]
check_conditions(run.events, [args.pos, args.neg])
run.events = run.events[run.events["trial_type"].isin([args.pos, args.neg])].reset_index(drop=True)
Y = subject_masker(run, args.fwhm).transform(run.img)

rows = []
for hrf, drift, noise in itertools.product(["boxcar", "spm", "spm+deriv"],
                                           ["none", "polynomial", "cosine"],
                                           ["ols", "ar1"]):
    X, res = fit_run(Y, run, [args.pos, args.neg], hrf, drift, noise)
    eff, var = run_contrast(res, X, {args.pos: 1.0, args.neg: -1.0})
    t = t_stat(eff, var)
    p = t_to_p(t, res.dof)
    rows.append(dict(hrf=hrf, drift=drift, noise=noise, n_regressors=X.shape[1], dof=res.dof,
                     n_fdr05=int(fdr_bh(p, 0.05)[0].sum()), n_p001=int((p < 1e-3).sum()),
                     max_t=float(t.max()), mean_abs_resid_lag1=float(np.mean(np.abs(res.lag1)))))
df = pd.DataFrame(rows)
df.to_csv(out / "design_ablation.csv", index=False)
print(df.to_string(index=False))
print("\nNOTE: more significant voxels is NOT automatically better; unmodelled "
      "autocorrelation inflates t. See 04_null_calibration.py for false-positive control.")

df["variant"] = df["hrf"] + "|" + df["drift"] + "|" + df["noise"]
fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
axes[0].bar(df["variant"], df["n_fdr05"]); axes[0].set_ylabel("# voxels (FDR q<.05)")
axes[1].bar(df["variant"], df["mean_abs_resid_lag1"], color="C1")
axes[1].set_ylabel("mean |lag-1 autocorr| of residuals")
plt.setp(axes[1].get_xticklabels(), rotation=90, fontsize=7)
fig.tight_layout(); fig.savefig(out / "design_ablation.png", dpi=150)

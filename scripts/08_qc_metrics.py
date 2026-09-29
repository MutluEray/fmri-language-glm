"""Per-subject QC metrics, related to first-level false-positive inflation.

Metrics (all from the preprocessed data + confounds file):
  - framewise displacement (Power et al. 2012 style: |d translation| (mm) + 50 mm * |d rotation (rad)|)
  - DVARS (RMS of the scan-to-scan signal change over the brain mask), reported as % of mean intensity
  - temporal SNR (median over voxels of mean/std over time)
  - median lag-1 autocorrelation of OLS residuals
  - residual power near the task fundamental frequency relative to high-frequency power (exploratory)

If results/null_quarter_shift.csv exists (script 07), each subject's median false-positive rate is merged in
and Spearman correlations are computed. With n = 10 subjects this is EXPLORATORY: report it as such.

Assumptions to verify: translations are in mm; rotation units are auto-detected (or set --rot-units).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from fmriglm.data import get_runs, subject_masker
from fmriglm.glm import lag1_autocorr
from fmriglm.pipeline import add_common_args, check_conditions, fit_run

NAME_SETS = [(["X", "Y", "Z"], ["RotX", "RotY", "RotZ"]),
             (["trans_x", "trans_y", "trans_z"], ["rot_x", "rot_y", "rot_z"])]

ap = add_common_args(argparse.ArgumentParser())
ap.add_argument("--rot-units", default="auto", choices=["auto", "rad", "deg"])
ap.add_argument("--fd-thresh", type=float, default=0.5, help="mm; scans above are counted as high-motion")
args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

runs = get_runs(args.data_dir)
conds = [args.pos, args.neg]

# ---- motion parameter names and units ----
cols = next(((t, r) for t, r in NAME_SETS if runs[0].confounds is not None
             and all(c in runs[0].confounds.columns for c in t + r)), None)
if cols is None:
    raise SystemExit("Could not find motion columns in the confounds file.")
trans, rot = cols
allrot = np.concatenate([r.confounds[rot].fillna(0).to_numpy(float).ravel() for r in runs])
alltr = np.concatenate([r.confounds[trans].fillna(0).to_numpy(float).ravel() for r in runs])
print(f"motion columns: translations {trans}, rotations {rot}")
print(f"pooled |translation| max = {np.abs(alltr).max():.3f}; pooled |rotation| max = {np.abs(allrot).max():.4f}")
units = args.rot_units
if units == "auto":
    units = "deg" if np.abs(allrot).max() > 0.5 else "rad"
    print(f"auto-detected rotation units: {units}  (rad if pooled max <= 0.5; VERIFY, or pass --rot-units)")
rot_scale = 50.0 * (np.pi / 180.0 if units == "deg" else 1.0)

rows = []
for run in runs:
    check_conditions(run.events, conds)
    ev = run.events[run.events["trial_type"].isin(conds)].reset_index(drop=True)
    Y = subject_masker(run, None).transform(run.img)               # (T, V), unsmoothed
    T = Y.shape[0]
    conf = run.confounds.fillna(0)

    dt = np.abs(np.diff(conf[trans].to_numpy(float), axis=0)).sum(1)
    dr = np.abs(np.diff(conf[rot].to_numpy(float), axis=0)).sum(1) * rot_scale
    fd = np.concatenate([[0.0], dt + dr])

    d = np.diff(Y, axis=0)
    dvars_pct = 100.0 * np.sqrt((d ** 2).mean(1)) / Y.mean()

    mu, sd = Y.mean(0), Y.std(0, ddof=1)
    ok = sd > 0
    tsnr = np.median(mu[ok] / sd[ok])

    X, _ = fit_run(Y, run, conds, args.hrf, args.drift, "ols", events=ev)
    Xn = X.to_numpy()
    resid = Y - Xn @ np.linalg.lstsq(Xn, Y, rcond=None)[0]
    lag1 = float(np.median(lag1_autocorr(resid)))

    period = float(np.median(np.diff(np.sort(ev.loc[ev["trial_type"] == args.pos, "onset"].to_numpy()))))
    f = np.fft.rfftfreq(T, d=run.tr)
    P = (np.abs(np.fft.rfft(resid - resid.mean(0), axis=0)) ** 2).mean(1)
    f0 = 1.0 / period
    band = (f >= f0 - 0.005) & (f <= f0 + 0.005)
    ratio = float(P[band].mean() / P[f >= 0.1].mean())

    rows.append(dict(subject=run.subject, n_scans=T, mean_fd=float(fd.mean()), max_fd=float(fd.max()),
                     n_fd_gt_thresh=int((fd > args.fd_thresh).sum()),
                     max_abs_trans=float(np.abs(conf[trans].to_numpy(float)).max()),
                     max_abs_rot_raw=float(np.abs(conf[rot].to_numpy(float)).max()),
                     dvars_median_pct=float(np.median(dvars_pct)), tsnr_median=float(tsnr),
                     lag1_median=lag1, task_freq_hz=f0, lowfreq_ratio=ratio))
    print(f"subject {run.subject}: meanFD={fd.mean():.3f} mm, tSNR={tsnr:.1f}, lag1={lag1:.3f}")

df = pd.DataFrame(rows)

nq = out / "null_quarter_shift.csv"
if nq.exists():
    d = pd.read_csv(nq, dtype={"subject": str})
    s = d.groupby(["subject", "noise"])[["fpr_p05", "fpr_p001"]].median().unstack()
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    df = df.merge(s, left_on="subject", right_index=True, how="left")
else:
    print(f"\n[note] {nq} not found; run scripts/07_null_quarter_shift.py first to relate QC to inflation.")

df.to_csv(out / "qc_metrics.csv", index=False)
pd.set_option("display.width", 200)
print("\n", df.round(4).to_string(index=False))

preds = ["mean_fd", "tsnr_median", "dvars_median_pct", "lag1_median", "lowfreq_ratio"]
outs = [c for c in ["fpr_p05_ols", "fpr_p05_ar1"] if c in df.columns]
if outs:
    print("\nSpearman correlations (n=%d, EXPLORATORY; p-values uncorrected):" % len(df))
    fig, axes = plt.subplots(len(outs), len(preds), figsize=(3.2 * len(preds), 3 * len(outs)), squeeze=False)
    for i, o in enumerate(outs):
        for j, p in enumerate(preds):
            rho, pv = spearmanr(df[p], df[o])
            print(f"  {o:12s} vs {p:18s}: rho={rho:+.2f}, p={pv:.3f}")
            ax = axes[i][j]
            ax.scatter(df[p], df[o])
            for _, r in df.iterrows():
                ax.annotate(r["subject"], (r[p], r[o]), fontsize=7)
            ax.axhline(0.05, color="k", ls="--", lw=0.8)
            ax.set_xlabel(p, fontsize=8); ax.set_ylabel(o, fontsize=8)
            ax.set_title(f"rho={rho:+.2f}", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "qc_vs_inflation.png", dpi=150)

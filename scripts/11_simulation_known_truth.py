"""A3: validation against known truth with simulated / surrogate noise and an injected effect.

For each subject the TRUE design is used (real events, canonical HRF, cosine drift, motion regressors).
A known effect (--effect-sd noise SDs, on the language regressor) is injected into --frac-active of voxels;
the remaining voxels are pure noise, so their rejection rate is the false-positive rate. Noise types:
  white           iid Gaussian with each voxel's real residual SD               -> calibrated if the code is right
  ar1             synthetic AR(1), rho = --rho                                  -> OLS should inflate, AR(1) should fix
  surrogate_full  real residuals of the FULL model (task regressors fitted out), phase-randomised per voxel
                  (biased low near the task frequency by construction: the task fit removed power there)
  surrogate_nuis  real residuals of a NUISANCE-ONLY model (drift, motion, constant), phase-randomised per voxel;
                  keeps the real noise spectrum near the task frequency. Voxels with |t| >= 3.1 under the true
                  design are excluded from all noise types so task-locked signal does not contaminate the spectrum.
Models: ols, ar1, ar1_bc (bias-corrected AR(1); see fmriglm.glm). Reported (two-sided; nominal 0.05 / 0.001):
FPR among inactive voxels, power (p<.001) among active voxels (only comparable when FPR is nominal), and
relative bias of the estimated contrast. If results/null_quarter_shift*.csv exist, surrogate FPRs are compared
(Spearman, n = 10, exploratory) with each subject's REAL quarter-shift FPR (different design: compare rank, not level).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from scipy.stats import spearmanr

from fmriglm.data import get_runs, subject_masker
from fmriglm.design import build_design, make_contrast
from fmriglm.glm import fit_glm, contrast, t_stat
from fmriglm.pipeline import add_common_args, check_conditions
from fmriglm.stats import t_to_p


def phase_randomize(R, rng):
    """Independent random phases per voxel; preserves each voxel's amplitude spectrum."""
    T = R.shape[0]
    F = np.fft.rfft(R - R.mean(0), axis=0)
    ph = rng.uniform(0.0, 2 * np.pi, size=F.shape)
    ph[0] = 0.0
    if T % 2 == 0:
        ph[-1] = 0.0
    return np.fft.irfft(np.abs(F) * np.exp(1j * ph), n=T, axis=0)


ap = add_common_args(argparse.ArgumentParser())
ap.add_argument("--n-rep", type=int, default=3, help="repetitions per surrogate type")
ap.add_argument("--effect-sd", type=float, default=0.5, help="injected effect in units of the voxel noise SD")
ap.add_argument("--frac-active", type=float, default=0.2)
ap.add_argument("--rho", type=float, default=0.4)
ap.add_argument("--models", nargs="+", default=["ols", "ar1", "ar1_bc"])
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(args.seed)

runs = get_runs(args.data_dir)
conds, w = [args.pos, args.neg], {args.pos: 1.0, args.neg: -1.0}
rows = []
for run in runs:
    check_conditions(run.events, conds)
    ev = run.events[run.events["trial_type"].isin(conds)].reset_index(drop=True)
    Y = subject_masker(run, None).transform(run.img)
    T = Y.shape[0]
    X = build_design(ev, conds, T, run.tr, args.hrf, args.drift, confounds=run.confounds, slice_ref=run.slice_ref)
    Xn = X.to_numpy()
    c = make_contrast(list(X.columns), w)
    nuis_cols = [k for k in X.columns if not any(k == q or k.startswith(q + "_derivative") for q in conds)]
    Xnu = X[nuis_cols].to_numpy()
    r0 = fit_glm(Y, Xn, "ols")
    keep = np.abs(t_stat(*contrast(r0, c))) < 3.1
    R_full = (Y - Xn @ np.linalg.lstsq(Xn, Y, rcond=None)[0])[:, keep]
    R_nuis = (Y - Xnu @ np.linalg.lstsq(Xnu, Y, rcond=None)[0])[:, keep]
    V = int(keep.sum())
    sd = R_full.std(0)
    x_pos = X[args.pos].to_numpy()
    print(f"subject {run.subject}: {V} voxels kept ({int((~keep).sum())} task-responsive voxels excluded)")
    for noise_type, n_rep in (("white", 1), ("ar1", 1), ("surrogate_full", args.n_rep), ("surrogate_nuis", args.n_rep)):
        for rep in range(n_rep):
            if noise_type == "white":
                N = rng.normal(size=(T, V)) * sd
            elif noise_type == "ar1":
                N = lfilter([1.0], [1.0, -args.rho], rng.normal(size=(T + 200, V)), axis=0)[200:]
                N = N / N.std(0) * sd
            elif noise_type == "surrogate_full":
                N = phase_randomize(R_full, rng)
            else:
                N = phase_randomize(R_nuis, rng)
            act = rng.random(V) < args.frac_active
            amp = np.where(act, args.effect_sd * N.std(0), 0.0)
            D = N + np.outer(x_pos, amp)
            for model in args.models:
                res = fit_glm(D, Xn, model)
                eff, var = contrast(res, c)
                p = t_to_p(t_stat(eff, var), res.dof, "two-sided")
                rows.append(dict(subject=run.subject, noise=noise_type, model=model, rep=rep,
                                 fpr_p05=float((p[~act] < 0.05).mean()), fpr_p001=float((p[~act] < 0.001).mean()),
                                 power_p001=float((p[act] < 0.001).mean()),
                                 rel_bias=float(np.mean(eff[act] / amp[act]) - 1.0)))

df = pd.DataFrame(rows)
df.to_csv(out / "simulation_known_truth.csv", index=False)
per = df.groupby(["noise", "model", "subject"])[["fpr_p05", "fpr_p001", "power_p001", "rel_bias"]].mean().reset_index()
summ = per.groupby(["noise", "model"])[["fpr_p05", "fpr_p001", "power_p001", "rel_bias"]].agg(["mean", "min", "max"])
pd.set_option("display.width", 240)
print("\nAcross subjects (nominal FPR: p05 = 0.05, p001 = 0.001):\n", summ.round(4).to_string())

real_file = next((out / f for f in ("null_quarter_shift_models.csv", "null_quarter_shift.csv") if (out / f).exists()), None)
if real_file is not None:
    real = pd.read_csv(real_file, dtype={"subject": str}).groupby(["subject", "noise"])["fpr_p05"].median().unstack()
    common = [m for m in args.models if m in real.columns]
    print(f"\nREAL quarter-shift FPR (p05) from {real_file.name}; models available: {list(real.columns)}")
    fig, axes = plt.subplots(len(common), 2, figsize=(9, 4 * len(common)), squeeze=False)
    for i, m in enumerate(common):
        for j, st in enumerate(["surrogate_full", "surrogate_nuis"]):
            sur = per[(per["noise"] == st) & (per["model"] == m)].set_index("subject")["fpr_p05"]
            d = pd.concat([sur.rename("surrogate"), real[m].rename("real")], axis=1).dropna()
            rho, pv = spearmanr(d["real"], d["surrogate"])
            print(f"  {m:7s} {st:15s}: mean surrogate FPR {d['surrogate'].mean():.4f} vs mean real {d['real'].mean():.4f}; "
                  f"Spearman rho = {rho:+.2f} (p = {pv:.3f}, n={len(d)}, exploratory)")
            ax = axes[i][j]; ax.scatter(d["real"], d["surrogate"])
            for s, r in d.iterrows():
                ax.annotate(s, (r["real"], r["surrogate"]), fontsize=7)
            ax.axhline(0.05, color="k", ls="--", lw=0.8); ax.axvline(0.05, color="k", ls="--", lw=0.8)
            ax.set_xlabel(f"real quarter-shift FPR ({m})"); ax.set_ylabel(f"{st} FPR"); ax.set_title(f"rho={rho:+.2f}", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "surrogate_vs_real_fpr.png", dpi=150)

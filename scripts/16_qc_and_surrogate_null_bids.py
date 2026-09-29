"""C1.4: motion QC and a surrogate-noise null for one fMRIPrep-processed BIDS run.

1. Motion QC from the fMRIPrep confounds: framewise displacement (mean/max, scans above 0.5/1 mm) and how strongly FD
   and each motion parameter correlate with the task regressor (task-locked motion is a classic source of false activation).
2. Observed t-map summaries per contrast and model: robust SD of t (1.4826 x MAD; ~1 if calibrated), fraction of voxels
   with t > 3 and t > 4.
3. Surrogate null: nuisance-only residuals (drift, motion, non-steady-state, optional aCompCor; NO task regressors) of a random
   sample of voxels that are not clearly task-responsive (max |t| over contrasts < 3.1), phase-randomised per voxel (keeps each
   voxel's noise spectrum, destroys task alignment), then fitted with the REAL design. Reports the null robust SD, fractions of t > 3
   and t > 4, and two-sided false-positive rates at p<.05 / .001, to compare with the observed values.

Caveats: excluding responsive voxels removes some noise-power near the task frequency (a small conservative bias); surrogate voxels are
independent (spatial correlation does not matter for per-voxel rates); one subject, one run per task.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import numpy as np
import pandas as pd
import nibabel as nib

from fmriglm.design import build_design, make_contrast
from fmriglm.glm import fit_glm, contrast, t_stat
from fmriglm.stats import t_to_p

CONTRASTS = {
    "fingerfootlips": {
        "Finger>others": {"Finger": 1.0, "Foot": -0.5, "Lips": -0.5},
        "Foot>others": {"Foot": 1.0, "Finger": -0.5, "Lips": -0.5},
        "Lips>others": {"Lips": 1.0, "Finger": -0.5, "Foot": -0.5},
        "AllTask>rest": {"Finger": 1 / 3, "Foot": 1 / 3, "Lips": 1 / 3},
    },
    "covertverbgeneration": {"Task>rest": {"Task": 1.0}},
}


def first(folder, patterns):
    for pat in patterns:
        hits = sorted(Path(folder).glob(pat))
        if hits:
            return hits[0]
    return None


def phase_randomize(R, rng):
    T = R.shape[0]
    F = np.fft.rfft(R - R.mean(0), axis=0)
    ph = rng.uniform(0.0, 2 * np.pi, size=F.shape)
    ph[0] = 0.0
    if T % 2 == 0:
        ph[-1] = 0.0
    return np.fft.irfft(np.abs(F) * np.exp(1j * ph), n=T, axis=0)


def robust_sd(t):
    return 1.4826 * float(np.median(np.abs(t - np.median(t))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bids", default="data/ds000114_subset")
    ap.add_argument("--deriv", default="data/derivatives")
    ap.add_argument("--sub", default="01")
    ap.add_argument("--ses", default="test")
    ap.add_argument("--task", default="covertverbgeneration", choices=list(CONTRASTS))
    ap.add_argument("--models", nargs="+", default=["ols", "ar1_bc"])
    ap.add_argument("--acompcor", type=int, default=0)
    ap.add_argument("--slice-ref", type=float, default=0.5)
    ap.add_argument("--n-vox", type=int, default=30000, help="voxels sampled for the surrogate null")
    ap.add_argument("--n-rep", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out) / "glm_bids"; out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    from nilearn.maskers import NiftiMasker

    task = args.task
    ev = pd.read_csv(Path(args.bids) / f"task-{task}_events.tsv", sep="\t")
    conds = sorted(ev["trial_type"].unique())
    fdir = Path(args.deriv) / f"sub-{args.sub}" / f"ses-{args.ses}" / "func"
    bold = first(fdir, [f"*task-{task}_*space-MNI152NLin2009cAsym*desc-preproc_bold.nii.gz"])
    mask = first(fdir, [f"*task-{task}_*space-MNI152NLin2009cAsym*desc-brain_mask.nii.gz"])
    cf = first(fdir, [f"*task-{task}_*desc-confounds_timeseries.tsv"])
    if bold is None or mask is None or cf is None:
        raise SystemExit(f"fMRIPrep outputs not found in {fdir}")
    conf = pd.read_csv(cf, sep="\t")
    cols = [c for c in ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"] if c in conf]
    cols += [c for c in conf.columns if c.startswith("non_steady_state_outlier")]
    cols += [f"a_comp_cor_{i:02d}" for i in range(args.acompcor) if f"a_comp_cor_{i:02d}" in conf]
    tr = float(nib.load(str(bold)).header.get_zooms()[3])
    Y = NiftiMasker(mask_img=str(mask), standardize=False).fit().transform(str(bold))
    T = Y.shape[0]
    X = build_design(ev, conds, T, tr, "spm", "cosine", confounds=conf, confound_cols=cols, slice_ref=args.slice_ref)
    Xn = X.to_numpy()
    print(f"{task}: {T} volumes, TR {tr}, {Y.shape[1]} voxels; design {X.shape}; nuisance {cols}")

    # ---- 1. motion QC ----
    fd = conf["framewise_displacement"].fillna(0).to_numpy()
    alltask = X[conds].sum(axis=1).to_numpy()
    mp = [c for c in ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"] if c in conf]
    cm = {c: np.corrcoef(conf[c].fillna(0).to_numpy(), alltask)[0, 1] for c in mp}
    print(f"\nMOTION QC: mean FD {fd.mean():.3f} mm, max {fd.max():.2f}, scans > 0.5 mm: {(fd > 0.5).sum()}, > 1 mm: {(fd > 1).sum()}")
    print(f"  corr(FD, task regressor) = {np.corrcoef(fd, alltask)[0, 1]:+.2f}; "
          f"largest |corr(motion parameter, task regressor)| = {max(abs(v) for v in cm.values()):.2f} "
          f"({max(cm, key=lambda k: abs(cm[k]))}); all: " + ", ".join(f"{k} {v:+.2f}" for k, v in cm.items()))

    ac = [c for c in cols if c.startswith("a_comp_cor")]
    if ac:
        print("  aCompCor components vs task regressor: " + ", ".join(
            f"{c[-2:]} {np.corrcoef(conf[c].fillna(0).to_numpy(), alltask)[0, 1]:+.2f}" for c in ac))

    cdict = CONTRASTS[task]
    cvec = {n: make_contrast(list(X.columns), w) for n, w in cdict.items()}

    # ---- 2. observed ----
    obs = {}
    for m in args.models:
        res = fit_glm(Y, Xn, m)
        for n, c in cvec.items():
            obs[(n, m)] = t_stat(*contrast(res, c))
        if m == args.models[0]:
            res0 = res
    tmax = np.max([np.abs(obs[(n, args.models[0])]) for n in cvec], axis=0)

    # ---- 3. surrogate null ----
    nuis = [c for c in X.columns if c not in conds]
    Xnu = X[nuis].to_numpy()
    keep = np.flatnonzero(tmax < 3.1)
    sel = rng.choice(keep, size=min(args.n_vox, keep.size), replace=False)
    R = Y[:, sel] - Xnu @ np.linalg.lstsq(Xnu, Y[:, sel], rcond=None)[0]
    print(f"\nSURROGATE NULL: {sel.size} voxels sampled from {keep.size} not clearly task-responsive "
          f"({Y.shape[1] - keep.size} responsive voxels excluded), {args.n_rep} repetitions")
    rows = []
    for rep in range(args.n_rep):
        Nz = phase_randomize(R, rng)
        for m in args.models:
            res = fit_glm(Nz, Xn, m)
            for n, c in cvec.items():
                t = t_stat(*contrast(res, c)); p2 = t_to_p(t, res.dof, "two-sided")
                rows.append(dict(contrast=n, model=m, rep=rep, null_robust_sd=robust_sd(t), null_frac_t_gt3=float((t > 3).mean()),
                                 null_frac_t_gt4=float((t > 4).mean()), null_fpr_p05=float((p2 < 0.05).mean()),
                                 null_fpr_p001=float((p2 < 0.001).mean())))
    nd = pd.DataFrame(rows).groupby(["contrast", "model"]).mean().drop(columns="rep").reset_index()
    nd["obs_robust_sd"] = [robust_sd(obs[(r.contrast, r.model)]) for r in nd.itertuples()]
    nd["obs_frac_t_gt3"] = [float((obs[(r.contrast, r.model)] > 3).mean()) for r in nd.itertuples()]
    nd["obs_frac_t_gt4"] = [float((obs[(r.contrast, r.model)] > 4).mean()) for r in nd.itertuples()]
    nd = nd[["contrast", "model", "obs_robust_sd", "null_robust_sd", "obs_frac_t_gt3", "null_frac_t_gt3",
             "obs_frac_t_gt4", "null_frac_t_gt4", "null_fpr_p05", "null_fpr_p001"]]
    pd.set_option("display.width", 220)
    print("\nObserved vs surrogate-null (robust SD ~1 if calibrated; null fractions are what noise alone gives for this design):\n",
          nd.round(4).to_string(index=False))
    nd.to_csv(out / f"surrogate_null_{task}.csv", index=False)


if __name__ == "__main__":
    main()

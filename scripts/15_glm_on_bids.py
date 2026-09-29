"""C1.3: run the from-scratch GLM on one BIDS run: fMRIPrep output (--mode fmriprep) or the RAW data (--mode raw).

fmriprep mode: MNI-space preprocessed BOLD + brain mask + confounds (6 motion parameters, non-steady-state outlier
               columns, optional --acompcor N) from <deriv>/sub-XX/ses-YY/func.
raw mode:      the unprocessed BOLD in native space, an EPI mask, NO motion/slice-timing/normalisation; the first
               --drop-initial volumes (default 4, the dataset's dummy scans) are dropped.
Both: canonical HRF, cosine drift (0.01 Hz), noise model --noise (default ar1_bc), events from the dataset-level
task-<task>_events.tsv. Contrasts: finger-foot-lips -> each condition vs the mean of the other two + all-task>rest;
covert verb generation -> task>rest (ONE condition against implicit rest: less language-specific than a
language-vs-control contrast).

Assumptions to check: files are found by glob patterns (printed); the slice-timing reference (--slice-ref, default 0.5)
is an assumption; comparisons between modes are descriptive (different spaces, voxel counts and dof).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage, stats

from fmriglm.design import build_design, make_contrast
from fmriglm.glm import fit_glm, contrast, t_stat
from fmriglm.mc import fdr_bh
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bids", default="data/ds000114_subset")
    ap.add_argument("--deriv", default="data/derivatives")
    ap.add_argument("--sub", default="01")
    ap.add_argument("--ses", default="test")
    ap.add_argument("--task", default="fingerfootlips", choices=list(CONTRASTS))
    ap.add_argument("--mode", default="fmriprep", choices=["fmriprep", "raw"])
    ap.add_argument("--noise", default="ar1_bc", choices=["ols", "ar1", "ar1_bc"])
    ap.add_argument("--fwhm", type=float, default=None)
    ap.add_argument("--drop-initial", type=int, default=None, help="default: 4 in raw mode, 0 in fmriprep mode")
    ap.add_argument("--acompcor", type=int, default=0, help="number of aCompCor components to add (fmriprep mode)")
    ap.add_argument("--slice-ref", type=float, default=0.5)
    ap.add_argument("--tr", type=float, default=None)
    ap.add_argument("--label-peaks", action="store_true", help="Harvard-Oxford labels for peaks (fmriprep mode; needs internet once)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out) / "glm_bids"; out.mkdir(parents=True, exist_ok=True)
    from nilearn.maskers import NiftiMasker
    from nilearn.masking import compute_epi_mask

    bids, deriv = Path(args.bids), Path(args.deriv)
    task = args.task
    ev = pd.read_csv(bids / f"task-{task}_events.tsv", sep="\t")
    conds = sorted(ev["trial_type"].unique())

    conf_df, cols, mask_img = None, None, None
    if args.mode == "fmriprep":
        fdir = deriv / f"sub-{args.sub}" / f"ses-{args.ses}" / "func"
        bold = first(fdir, [f"*task-{task}_*space-MNI152NLin2009cAsym*desc-preproc_bold.nii.gz",
                            f"*task-{task}_*desc-preproc_bold.nii.gz"])
        mask = first(fdir, [f"*task-{task}_*space-MNI152NLin2009cAsym*desc-brain_mask.nii.gz",
                            "*space-MNI152NLin2009cAsym*desc-brain_mask.nii.gz"])
        cf = first(fdir, [f"*task-{task}_*desc-confounds_timeseries.tsv"])
        if bold is None or cf is None:
            raise SystemExit(f"fMRIPrep outputs not found in {fdir}. Found: {sorted(p.name for p in fdir.glob('*'))[:20]}")
        print("BOLD    :", bold.name, "\nmask    :", mask.name if mask else None, "\nconfounds:", cf.name)
        conf_df = pd.read_csv(cf, sep="\t")
        cols = [c for c in ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"] if c in conf_df]
        cols += [c for c in conf_df.columns if c.startswith("non_steady_state_outlier")]
        cols += [f"a_comp_cor_{i:02d}" for i in range(args.acompcor) if f"a_comp_cor_{i:02d}" in conf_df]
        mask_img = str(mask) if mask else None
        n0 = 0 if args.drop_initial is None else args.drop_initial
    else:
        bold = first(bids / f"sub-{args.sub}" / f"ses-{args.ses}" / "func", [f"*task-{task}_bold.nii.gz"])
        if bold is None:
            raise SystemExit("raw BOLD not found")
        print("BOLD (raw):", bold.name)
        n0 = 4 if args.drop_initial is None else args.drop_initial

    hdr_tr = float(nib.load(str(bold)).header.get_zooms()[3])
    tr = args.tr or hdr_tr
    if mask_img is None:
        from nilearn.image import mean_img, new_img_like
        mask_img = compute_epi_mask(str(bold))
        if mask_img.get_fdata().sum() == 0:      # nilearn returned an empty mask: fall back to a simple intensity threshold
            print("[warn] EPI mask was empty; using mean image > 20% of its 98th percentile")
            mimg = mean_img(str(bold)); dat = mimg.get_fdata()
            mask_img = new_img_like(mimg, (dat > 0.2 * np.percentile(dat, 98)).astype(np.uint8))
    masker = NiftiMasker(mask_img=mask_img,
                         smoothing_fwhm=args.fwhm, standardize=False).fit()
    Y = masker.transform(str(bold))
    T0 = Y.shape[0]
    Y = Y[n0:]
    T = Y.shape[0]
    ev = ev.copy(); ev["onset"] = ev["onset"] - n0 * tr
    ev = ev[ev["onset"] >= 0].reset_index(drop=True)
    if conf_df is not None:
        conf_df = conf_df.iloc[n0:].reset_index(drop=True)
        cols = [c for c in cols if conf_df[c].fillna(0).abs().sum() > 0]
    mean_fd = float(conf_df["framewise_displacement"].mean()) if conf_df is not None and "framewise_displacement" in conf_df else float("nan")
    print(f"{args.mode}: {T0} volumes, dropped first {n0}, analysed {T}; TR={tr}; voxels={Y.shape[1]}; nuisance columns={cols}; "
          f"mean FD={mean_fd:.3f}")

    X = build_design(ev, conds, T, tr, "spm", "cosine", confounds=conf_df, confound_cols=cols, slice_ref=args.slice_ref)
    res = fit_glm(Y, X.to_numpy(), args.noise)
    print(f"design {X.shape}, dof={res.dof:.0f}, median residual lag-1 = {np.median(res.lag1):.3f}")

    mask3d = masker.mask_img_.get_fdata() > 0
    ijk = np.argwhere(mask3d)
    xyz = nib.affines.apply_affine(masker.mask_img_.affine, ijk)
    lab = None
    if args.label_peaks and args.mode == "fmriprep":
        from nilearn.datasets import fetch_atlas_harvard_oxford
        from nilearn.image import load_img, resample_to_img
        at = fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
        aimg = resample_to_img(load_img(at.maps), masker.mask_img_, interpolation="nearest")
        ad = np.rint(aimg.get_fdata()).astype(int)
        lab = (list(at.labels), ad[ijk[:, 0], ijk[:, 1], ijk[:, 2]])

    rows = []
    for cname, w in CONTRASTS[task].items():
        eff, var = contrast(res, make_contrast(list(X.columns), w))
        t = t_stat(eff, var)
        p = t_to_p(t, res.dof)
        sig, cut = fdr_bh(p, 0.05)
        robust_sd = 1.4826 * float(np.median(np.abs(t - np.median(t))))
        print(f"\n[{cname}] FDR q<.05: {int(sig.sum())} voxels | max t {t.max():.2f} | robust SD of t {robust_sd:.2f}")
        vol = np.zeros(mask3d.shape, int); vol[tuple(ijk.T)] = sig.astype(int)
        cc, n_cc = ndimage.label(vol, structure=ndimage.generate_binary_structure(3, 1))
        cvec = cc[tuple(ijk.T)]
        peaks = []
        for c in range(1, n_cc + 1):
            m = np.flatnonzero(cvec == c); pk = m[np.argmax(t[m])]
            name = ""
            if lab is not None:
                name = lab[0][lab[1][pk]] if lab[1][pk] > 0 else "(non-cortical / outside atlas)"
            peaks.append((float(t[pk]), len(m), tuple(int(v) for v in np.round(xyz[pk])), name))
        for k, (tv, n, xy, name) in enumerate(sorted(peaks, reverse=True)[:6]):
            print(f"   peak {k + 1}: t={tv:.2f}, cluster {n} vox, xyz={xy} {name}")
        rows.append(dict(mode=args.mode, sub=args.sub, ses=args.ses, task=task, contrast=cname, T=T, n_vox=Y.shape[1], dof=res.dof,
                         n_fdr05=int(sig.sum()), max_t=float(t.max()), robust_sd_t=robust_sd,
                         median_lag1=float(np.median(res.lag1)), mean_fd=mean_fd, n_nuisance=len(cols or []), fwhm=args.fwhm,
                         noise=args.noise, acompcor=args.acompcor))
        tag = (f"{args.mode}_{args.ses}_{args.noise}{'_ac' + str(args.acompcor) if args.acompcor else ''}_"
               f"{task}_{cname.replace('>', '_gt_')}")
        masker.inverse_transform(t).to_filename(str(out / f"{tag}_tmap.nii.gz"))
        masker.mask_img_.to_filename(str(out / f"{tag}_mask.nii.gz"))
        # NOTE: this plot uses nilearn's per-map defaults -- its OWN auto-picked slice
        # (wherever THIS map's peak happens to be) and its OWN auto-scaled colorbar max.
        # That is fine for viewing one run's own result, but comparing two of these PNGs
        # across runs/sessions/models side by side is misleading (different crop, different
        # color scale). For a fair comparison use scripts/19_compare_fixed_view.py, which
        # fixes cut_coords/threshold/vmax across the maps being compared.
        try:
            from nilearn import plotting
            from nilearn.image import mean_img
            thr = float(stats.t.isf(cut, res.dof)) if cut > 0 else 3.1
            kw = {"bg_img": mean_img(str(bold))} if args.mode == "raw" else {}
            plotting.plot_stat_map(masker.inverse_transform(t), threshold=thr,
                                   title=f"{args.mode}: {task} {cname}", output_file=str(out / f"{tag}.png"), **kw)
        except Exception as e:
            print(f"   [warn] plotting failed: {e!r}")
    summ = out / "glm_bids_summary.csv"
    df = pd.DataFrame(rows)
    df.to_csv(summ, mode="a", header=not summ.exists(), index=False)
    print(f"\nappended {len(df)} rows to {summ}")


if __name__ == "__main__":
    main()

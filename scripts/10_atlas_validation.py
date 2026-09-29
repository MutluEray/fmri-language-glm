"""A2: anatomical validation of the language-localizer result (Harvard-Oxford atlas).

PRE-SPECIFIED (written before looking at any atlas output):
  Primary analysis: FWHM 6 mm (--fwhm), AR(1), canonical HRF, cosine drift, contrast language > string.
  Language ROIs, left and right (Harvard-Oxford cortical, maxprob thr25 2 mm):
    IFG pars triangularis, IFG pars opercularis, STG anterior, STG posterior,
    MTG anterior, MTG posterior, Angular Gyrus.
  Control ROIs: Occipital Pole, Precentral Gyrus.
  All ROI x hemisphere results are reported; BH-FDR is applied across all of them.

Outputs (results/): atlas_peaks.csv, laterality.csv, roi_effects.csv, roi_effects.png.
Also prints the raw-units vs %-of-global-mean group comparison (script 05 used raw units).

Caveats: n = 10 subjects; the atlas is probabilistic-maxprob and the group grid is coarse (~4 mm); the atlas is
resampled with nearest-neighbour; hemisphere is defined by the sign of the MNI x coordinate (x<0 left);
laterality indices depend on the threshold and are crude; ROI tests are descriptive.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from scipy import ndimage, stats

from fmriglm.data import get_runs
from fmriglm.glm import t_stat
from fmriglm.mc import fdr_bh, sign_flip_maxT
from fmriglm.pipeline import add_common_args, check_conditions, fit_run, run_contrast
from fmriglm.stats import one_sample_t, t_to_p

LANG = ["Inferior Frontal Gyrus, pars triangularis", "Inferior Frontal Gyrus, pars opercularis",
        "Superior Temporal Gyrus, anterior division", "Superior Temporal Gyrus, posterior division",
        "Middle Temporal Gyrus, anterior division", "Middle Temporal Gyrus, posterior division",
        "Angular Gyrus"]
CONTROL = ["Occipital Pole", "Precentral Gyrus"]

ap = add_common_args(argparse.ArgumentParser())
ap.set_defaults(fwhm=6.0)
ap.add_argument("--n-perm", type=int, default=5000)
ap.add_argument("--top-frac", type=float, default=0.10, help="adaptive per-subject LI: top fraction of ROI voxels")
args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

from nilearn.masking import compute_epi_mask, intersect_masks
from nilearn.maskers import NiftiMasker
from nilearn.datasets import fetch_atlas_harvard_oxford
from nilearn.image import load_img, resample_to_img

runs = get_runs(args.data_dir)
conds, w = [args.pos, args.neg], {args.pos: 1.0, args.neg: -1.0}
for r in runs:
    check_conditions(r.events, conds)
    r.events = r.events[r.events["trial_type"].isin(conds)].reset_index(drop=True)
group_mask = intersect_masks([compute_epi_mask(r.img) for r in runs], threshold=1.0)
masker = NiftiMasker(mask_img=group_mask, smoothing_fwhm=args.fwhm, standardize=False).fit()

eff_raw, eff_psc, t_sub = [], [], []
for r in runs:
    Y = masker.transform(r.img)
    X, res = fit_run(Y, r, conds, args.hrf, args.drift, args.noise)
    e, v = run_contrast(res, X, w)
    eff_raw.append(e); eff_psc.append(100.0 * e / float(Y.mean())); t_sub.append(t_stat(e, v))
    print(f"subject {r.subject} fitted")
eff_raw, eff_psc, t_sub = np.vstack(eff_raw), np.vstack(eff_psc), np.vstack(t_sub)
S, V = eff_psc.shape


def group_stats(E):
    t, dof = one_sample_t(E)
    p = t_to_p(t, dof)
    sf = sign_flip_maxT(E, n_perm=args.n_perm)
    return dict(t=t, dof=dof, p=p, fdr=fdr_bh(p, 0.05)[0], p_fwe=sf["p_fwe"])


G_raw, G = group_stats(eff_raw), group_stats(eff_psc)
print(f"\nGroup result, {S} subjects x {V} voxels, FWHM {args.fwhm} mm")
print(f"  raw units          : FDR {int(G_raw['fdr'].sum())}, FWE {int((G_raw['p_fwe'] < .05).sum())}")
print(f"  % global mean units: FDR {int(G['fdr'].sum())}, FWE {int((G['p_fwe'] < .05).sum())}   <- used below")

# ---- atlas in the group grid ----
atlas = fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
aimg = load_img(atlas.maps)
try:
    aimg = resample_to_img(aimg, group_mask, interpolation="nearest", force_resample=True, copy_header=True)
except TypeError:
    aimg = resample_to_img(aimg, group_mask, interpolation="nearest")
adata = np.rint(aimg.get_fdata()).astype(int)
labels = list(atlas.labels)
mask3d = group_mask.get_fdata() > 0
ijk = np.argwhere(mask3d)                       # same (C) order as the masker's voxel vector
xyz = nib.affines.apply_affine(group_mask.affine, ijk)
lab = adata[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
left, right = xyz[:, 0] < 0, xyz[:, 0] > 0


def idx_of(name):
    return labels.index(name) if name in labels else None


missing = [n for n in LANG + CONTROL if idx_of(n) is None]
if missing:
    print("\n[warn] labels not found in this atlas version (skipped):", missing)
    print("       available:", labels)

# ---- peak table (FDR clusters; flag FWE) ----
vol = np.zeros(mask3d.shape, int)
vol[tuple(ijk.T)] = G["fdr"].astype(int)
cc, n_cc = ndimage.label(vol, structure=ndimage.generate_binary_structure(3, 1))
cc_vec = cc[tuple(ijk.T)]
rows = []
for c in range(1, n_cc + 1):
    m = np.flatnonzero(cc_vec == c)
    pk = m[np.argmax(G["t"][m])]
    rows.append(dict(cluster=c, n_vox=len(m), peak_t=round(float(G["t"][pk]), 2),
                     mni_x=round(float(xyz[pk, 0])), mni_y=round(float(xyz[pk, 1])), mni_z=round(float(xyz[pk, 2])),
                     atlas_label=labels[lab[pk]] if lab[pk] > 0 else "(non-cortical / outside atlas)",
                     n_fwe_vox=int((G["p_fwe"][m] < 0.05).sum())))
peaks = pd.DataFrame(rows).sort_values("peak_t", ascending=False) if rows else pd.DataFrame()
peaks.to_csv(out / "atlas_peaks.csv", index=False)
print("\nFDR clusters (q<.05) with peak labels:\n", peaks.to_string(index=False) if len(peaks) else "  none")

# ---- laterality ----
lang_idx = [i for i in (idx_of(n) for n in LANG) if i is not None]
masks = {"all cortical voxels": lab > 0, "language ROIs": np.isin(lab, lang_idx)}


def li(l, r_):
    return np.nan if (l + r_) == 0 else (l - r_) / (l + r_)


lat = []
for mname, mk in masks.items():
    for thr in (3.0, 4.0, 5.0):
        sel = mk & (G["t"] > thr)
        lat.append(dict(scope=f"group, {mname}", measure=f"voxels t>{thr}", L=int((sel & left).sum()),
                        R=int((sel & right).sum()), LI=li((sel & left).sum(), (sel & right).sum())))
for k in range(S):
    mk = masks["language ROIs"]
    cand = np.flatnonzero(mk)
    top = cand[np.argsort(t_sub[k][cand])[-max(1, int(args.top_frac * len(cand))):]]
    L_, R_ = int(left[top].sum()), int(right[top].sum())
    lat.append(dict(scope=f"subject {runs[k].subject}", measure=f"top {int(100 * args.top_frac)}% of language-ROI voxels",
                    L=L_, R=R_, LI=li(L_, R_)))
lat = pd.DataFrame(lat)
lat.to_csv(out / "laterality.csv", index=False)
print("\nLaterality (LI = (L-R)/(L+R); +1 = fully left):\n", lat.round(3).to_string(index=False))
subj_li = lat[lat["scope"].str.startswith("subject")]["LI"]
print(f"\nper-subject LI: median {subj_li.median():.2f}; left-lateralised (LI>0) in {(subj_li > 0).sum()}/{len(subj_li)}")

# ---- ROI analysis ----
res_rows = []
for name in LANG + CONTROL:
    i = idx_of(name)
    if i is None:
        continue
    per = {}
    for hemi, hm in (("L", left), ("R", right)):
        sel = (lab == i) & hm
        per[hemi] = eff_psc[:, sel].mean(1) if sel.any() else np.full(S, np.nan)
        m_, sd = np.nanmean(per[hemi]), np.nanstd(per[hemi], ddof=1)
        tt, pp = stats.ttest_1samp(per[hemi], 0.0, nan_policy="omit")
        ci = stats.t.ppf(0.975, S - 1) * sd / np.sqrt(S)
        res_rows.append(dict(roi=name, hemi=hemi, kind="language" if name in LANG else "control",
                             n_vox=int(sel.sum()), mean_pct=m_, ci95=ci, t=tt, p=pp, n_pos=int((per[hemi] > 0).sum())))
    d = per["L"] - per["R"]
    tt, pp = stats.ttest_1samp(d, 0.0, nan_policy="omit")
    res_rows.append(dict(roi=name, hemi="L-R", kind="language" if name in LANG else "control", n_vox=np.nan,
                         mean_pct=np.nanmean(d), ci95=stats.t.ppf(0.975, S - 1) * np.nanstd(d, ddof=1) / np.sqrt(S),
                         t=tt, p=pp, n_pos=int((d > 0).sum())))
roi = pd.DataFrame(res_rows)
roi["p_fdr_bh_significant"] = fdr_bh(roi["p"].fillna(1.0).to_numpy(), 0.05)[0]
roi.to_csv(out / "roi_effects.csv", index=False)
print("\nROI effects (% of global mean intensity; language > string), one-sample t across subjects vs 0 "
      "(L-R rows: paired difference). p uncorrected; last column = BH-FDR across all rows:\n",
      roi.round(4).to_string(index=False))

sub = roi[roi["hemi"].isin(["L", "R"])]
fig, ax = plt.subplots(figsize=(8, 5))
names = list(dict.fromkeys(sub["roi"]))
y = np.arange(len(names))
for off, hemi in ((-0.18, "L"), (0.18, "R")):
    d = sub[sub["hemi"] == hemi].set_index("roi").loc[names]
    ax.errorbar(d["mean_pct"], y + off, xerr=d["ci95"], fmt="o", label=hemi, capsize=2)
ax.axvline(0, color="k", lw=0.8)
ax.set_yticks(y); ax.set_yticklabels([n.replace("division", "div.") for n in names], fontsize=7)
ax.set_xlabel("language - string effect (% of global mean), mean +/- 95% CI over subjects"); ax.legend()
fig.tight_layout(); fig.savefig(out / "roi_effects.png", dpi=150)

"""Group-level inference: parametric vs FDR vs Bonferroni vs sign-flip max-T (FWE) vs cluster extent."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import numpy as np
import pandas as pd
from scipy import stats

from fmriglm.data import get_runs
from fmriglm.pipeline import add_common_args, check_conditions, fit_run, run_contrast
from fmriglm.stats import combine_fixed, one_sample_t, t_to_p
from fmriglm.mc import fdr_bh, bonferroni, sign_flip_maxT, cluster_extent
from fmriglm.viz import save_stat_map

ap = add_common_args(argparse.ArgumentParser())
ap.add_argument("--n-perm", type=int, default=5000)
ap.add_argument("--cluster-p", type=float, default=0.001, help="cluster-forming p")
ap.add_argument("--min-cluster", type=int, default=10, help="min cluster size (voxels)")
args = ap.parse_args()
out = Path(args.out); out.mkdir(exist_ok=True)

from nilearn.masking import compute_epi_mask, intersect_masks
from nilearn.maskers import NiftiMasker

runs = get_runs(args.data_dir)
for r in runs:
    check_conditions(r.events, [args.pos, args.neg])
    r.events = r.events[r.events["trial_type"].isin([args.pos, args.neg])].reset_index(drop=True)

group_mask = intersect_masks([compute_epi_mask(r.img) for r in runs], threshold=1.0)
masker = NiftiMasker(mask_img=group_mask, smoothing_fwhm=args.fwhm, standardize=False).fit()

by_sub = {}
for r in runs:
    by_sub.setdefault(r.subject, []).append(r)

effects = []
for sub, rs in by_sub.items():
    effs, vars_ = [], []
    for r in rs:
        Y = masker.transform(r.img)
        X, res = fit_run(Y, r, [args.pos, args.neg], args.hrf, args.drift, args.noise)
        e, v = run_contrast(res, X, {args.pos: 1.0, args.neg: -1.0})
        effs.append(e); vars_.append(v)
    effects.append(combine_fixed(effs, vars_)[0])
    print(f"subject {sub}: {len(rs)} run(s) fitted")
effects = np.vstack(effects)
S, V = effects.shape
print(f"\ngroup effects: {S} subjects x {V} voxels")

t, dof = one_sample_t(effects)
p = t_to_p(t, dof)
fdr_mask, fdr_cut = fdr_bh(p, 0.05)
bonf_mask = bonferroni(p, 0.05)
sf = sign_flip_maxT(effects, n_perm=args.n_perm)
fwe_mask = sf["p_fwe"] < 0.05

cdt = t_to_p(np.inf, dof)  # placeholder to keep name defined
cluster_vec = (p < args.cluster_p).astype(float)
vol = masker.inverse_transform(cluster_vec).get_fdata() > 0.5
kept = cluster_extent(vol, args.min_cluster)
from nilearn.image import new_img_like
kept_img = new_img_like(group_mask, kept.astype(np.float32))
cluster_mask = masker.transform(kept_img).ravel() > 0.5

rows = [
    ("uncorrected p<.001", int((p < 0.001).sum())),
    ("FDR q<.05 (BH)", int(fdr_mask.sum())),
    ("Bonferroni p<.05", int(bonf_mask.sum())),
    (f"sign-flip max-T FWE p<.05 ({'exact' if sf['exact'] else str(args.n_perm) + ' perms'})", int(fwe_mask.sum())),
    (f"cluster p<{args.cluster_p} & >= {args.min_cluster} vox (heuristic)", int(cluster_mask.sum())),
]
df = pd.DataFrame(rows, columns=["method", "n_significant_voxels"])
df.to_csv(out / "group_inference.csv", index=False)
print(f"\n{S} subjects; dof={dof}\n", df.to_string(index=False))
if sf["exact"]:
    print(f"(all {2 ** S} sign patterns enumerated; smallest attainable FWE p = {1 / 2 ** S:.4f})")

masker.inverse_transform(t).to_filename(str(out / "group_t_map.nii.gz"))
for name, m in [("fdr", fdr_mask), ("fwe", fwe_mask), ("cluster", cluster_mask)]:
    if m.any():
        save_stat_map(masker.inverse_transform(t * m), out / f"group_{name}.png",
                      threshold=1e-6, title=f"group {args.pos}>{args.neg}: {name}")

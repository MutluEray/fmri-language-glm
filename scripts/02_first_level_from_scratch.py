"""Single-run GLM from scratch, validated against nilearn's FirstLevelModel."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import numpy as np
import matplotlib.pyplot as plt

from fmriglm.data import get_runs, subject_masker
from fmriglm.design import MOTION_COLS
from fmriglm.pipeline import add_common_args, check_conditions, fit_run, run_contrast
from fmriglm.stats import t_to_p
from fmriglm.glm import t_stat
from fmriglm.mc import fdr_bh
from fmriglm.viz import plot_design, save_stat_map

ap = add_common_args(argparse.ArgumentParser())
ap.add_argument("--run-index", type=int, default=0)
args = ap.parse_args()
out = Path(args.out); out.mkdir(exist_ok=True)

runs = get_runs(args.data_dir)
run = runs[args.run_index]
check_conditions(run.events, [args.pos, args.neg])
run.events = run.events[run.events["trial_type"].isin([args.pos, args.neg])].reset_index(drop=True)
masker = subject_masker(run, args.fwhm)
Y = masker.transform(run.img)
print(f"subject {run.subject}: Y = {Y.shape} (scans x voxels)")

X, res = fit_run(Y, run, [args.pos, args.neg], args.hrf, args.drift, args.noise)
plot_design(X, out / "design_matrix.png", f"{args.hrf} / {args.drift}")
eff, var = run_contrast(res, X, {args.pos: 1.0, args.neg: -1.0})
t = t_stat(eff, var)
p = t_to_p(t, res.dof)
sig, cut = fdr_bh(p, 0.05)
print(f"dof={res.dof:.0f}; voxels FDR q<.05: {sig.sum()} (p cutoff {cut:.2e}); max t={t.max():.2f}")
save_stat_map(masker.inverse_transform(t), out / "t_map_ours.png", threshold=3.1,
              title=f"ours: {args.pos} > {args.neg} ({args.noise})")

# ---- validation against nilearn (same HRF/drift/noise family) ----
from nilearn.glm.first_level import FirstLevelModel
cols = [c for c in MOTION_COLS if run.confounds is not None and c in run.confounds.columns]
conf = run.confounds[cols].fillna(0) if cols else None
fl = FirstLevelModel(t_r=run.tr, slice_time_ref=run.slice_ref, hrf_model="spm",
                     drift_model="cosine", high_pass=0.01, noise_model=args.noise,
                     smoothing_fwhm=args.fwhm, signal_scaling=False,
                     mask_img=masker.mask_img_, minimize_memory=True)
fl.fit(run.img, events=run.events, confounds=conf)
t_nl = masker.transform(fl.compute_contrast(f"{args.pos} - {args.neg}", output_type="stat")).ravel()
r = np.corrcoef(t, t_nl)[0, 1]
print(f"nilearn agreement (only meaningful for --hrf spm --drift cosine): "
      f"Pearson r={r:.5f}, mean|dt|={np.mean(np.abs(t - t_nl)):.4f}")

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(t_nl, t, s=2, alpha=0.3)
lim = [min(t.min(), t_nl.min()), max(t.max(), t_nl.max())]
ax.plot(lim, lim, "r--", lw=1)
ax.set_xlabel("nilearn t"); ax.set_ylabel("from-scratch t"); ax.set_title(f"r = {r:.4f}")
fig.tight_layout(); fig.savefig(out / "validation_vs_nilearn.png", dpi=150)

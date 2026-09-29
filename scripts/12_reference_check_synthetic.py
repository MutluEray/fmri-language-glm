"""A3c: reference-implementation check on SYNTHETIC data with known noise.

The same synthetic 4D images (white noise, or AR(1) noise with --rho) are analysed with nilearn's
FirstLevelModel (ols, ar1) and with our GLM (ols, ar1, ar1_bc), using a real subject's events, canonical HRF and
cosine drift, and no confounds (so the designs match). There is no task signal, so every rejection is a false
positive. Reported: two-sided false-positive rate at p<.05 and p<.001 (nominal 0.05 / 0.001).

Question answered: is the anticonservative behaviour of per-voxel AR(1) prewhitening specific to our code, or shared
with the reference implementation? Agreement with nilearn alone shows equivalence; this shows calibration against
known truth for BOTH.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.signal import lfilter
from scipy.stats import norm

from fmriglm.design import build_design, make_contrast
from fmriglm.glm import fit_glm, contrast, t_stat
from fmriglm.stats import t_to_p


def synth(noise, shape, T, rho, rng):
    if noise == "white":
        d = rng.normal(size=shape + (T,))
    else:
        d = lfilter([1.0], [1.0, -rho], rng.normal(size=shape + (T + 200,)), axis=-1)[..., 200:]
    return 100.0 + d


def nilearn_p(data, events, tr, slice_ref, noise_model, pos, neg):
    from nilearn.glm.first_level import FirstLevelModel
    img = nib.Nifti1Image(data.astype(np.float32), np.diag([3.0, 3.0, 3.0, 1.0]))
    mask = nib.Nifti1Image(np.ones(data.shape[:3], np.int8), img.affine)
    fl = FirstLevelModel(t_r=tr, slice_time_ref=slice_ref, hrf_model="spm", drift_model="cosine", high_pass=0.01,
                         noise_model=noise_model, signal_scaling=False, smoothing_fwhm=None, mask_img=mask,
                         minimize_memory=True)
    fl.fit(img, events=events)
    z = fl.compute_contrast(f"{pos} - {neg}", output_type="z_score").get_fdata().ravel()
    return 2 * norm.sf(np.abs(z))


def our_p(data, events, tr, slice_ref, model, pos, neg):
    T = data.shape[-1]
    Y = data.reshape(-1, T).T
    X = build_design(events, [pos, neg], T, tr, "spm", "cosine", slice_ref=slice_ref)
    res = fit_glm(Y, X.to_numpy(), model)
    c = make_contrast(list(X.columns), {pos: 1.0, neg: -1.0})
    return t_to_p(t_stat(*contrast(res, c)), res.dof, "two-sided")


def main():
    from fmriglm.data import get_runs
    from fmriglm.pipeline import add_common_args
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--run-index", type=int, default=0)
    ap.add_argument("--n-rep", type=int, default=5)
    ap.add_argument("--shape", type=int, nargs=3, default=[20, 20, 20])
    ap.add_argument("--rho", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    import nilearn
    print("nilearn version:", nilearn.__version__)

    run = get_runs(args.data_dir)[args.run_index]
    ev = run.events[run.events["trial_type"].isin([args.pos, args.neg])].reset_index(drop=True)
    T = int(round(ev["onset"].max() / run.tr)) + 40      # only used if the image length is unknown
    from nilearn.image import load_img
    T = load_img(run.img).shape[-1]
    rng = np.random.default_rng(args.seed)
    rows = []
    for noise in ("white", "ar1"):
        for rep in range(args.n_rep):
            data = synth(noise, tuple(args.shape), T, args.rho, rng)
            ps = {"nilearn_ols": nilearn_p(data, ev, run.tr, run.slice_ref, "ols", args.pos, args.neg),
                  "nilearn_ar1": nilearn_p(data, ev, run.tr, run.slice_ref, "ar1", args.pos, args.neg)}
            for m in ("ols", "ar1", "ar1_bc"):
                ps[f"ours_{m}"] = our_p(data, ev, run.tr, run.slice_ref, m, args.pos, args.neg)
            for name, p in ps.items():
                rows.append(dict(noise=noise, rep=rep, method=name, n_vox=p.size,
                                 fpr_p05=float((p < 0.05).mean()), fpr_p001=float((p < 0.001).mean())))
            print(f"{noise} rep {rep + 1}/{args.n_rep} done")
    df = pd.DataFrame(rows)
    df.to_csv(out / "reference_check_synthetic.csv", index=False)
    summ = df.groupby(["noise", "method"])[["fpr_p05", "fpr_p001"]].agg(["mean", "std"]).round(4)
    print(f"\nFalse-positive rates over {args.n_rep} reps x {np.prod(args.shape)} voxels "
          f"(nominal 0.05 / 0.001; binomial SE of the pooled p05 rate ~{np.sqrt(.05 * .95 / (args.n_rep * np.prod(args.shape))):.4f}):\n")
    print(summ.to_string())


if __name__ == "__main__":
    main()

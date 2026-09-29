"""Fixed-view comparison of two or more saved t-maps (e.g. test vs retest, or two noise
models): the SAME cut coordinates, SAME display threshold and SAME colorbar max for every
panel, so panels can be read side by side without being misled by nilearn's per-map
defaults (see the caveat in 15_glm_on_bids.py, which auto-picks each map's own peak
location and its own color range -- fine for viewing one run's own FDR result, actively
misleading for comparing runs).

Produces one combined figure (one row per map, shared cut/threshold/colorbar) plus one
PNG per map at the same settings, so panels can also be used individually.

Example (test vs retest, same contrast, same noise model):
  python3 scripts/19_compare_fixed_view.py \\
    --maps test=results/glm_bids/fmriprep_test_ar1_bc_fingerfootlips_Finger_gt_others_tmap.nii.gz \\
           retest=results/glm_bids/fmriprep_retest_ar1_bc_fingerfootlips_Finger_gt_others_tmap.nii.gz \\
    --threshold 3.1 --title "Finger>others"
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nilearn import plotting


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", nargs="+", required=True,
                    help="label=path.nii.gz pairs, e.g. test=a.nii.gz retest=b.nii.gz")
    ap.add_argument("--cut-coords", type=float, nargs=3, default=None,
                    help="shared x y z in mm; default: the peak |t| voxel of the FIRST map listed")
    ap.add_argument("--threshold", type=float, default=3.1, help="shared display threshold (default ~p<.001 uncorrected)")
    ap.add_argument("--vmax", type=float, default=None, help="shared colorbar max; default: max |t| across all maps")
    ap.add_argument("--title", default=None, help="prefix for each panel's title, e.g. 'Finger>others'")
    ap.add_argument("--out", default="results/glm_bids/compare")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    pairs = [m.split("=", 1) for m in args.maps]
    if any(len(p) != 2 for p in pairs):
        raise SystemExit("each --maps entry must be label=path.nii.gz")
    imgs = [nib.load(p) for _, p in pairs]

    if args.cut_coords is None:
        data = imgs[0].get_fdata()
        ijk = np.unravel_index(np.argmax(np.abs(data)), data.shape)
        cut = tuple(float(v) for v in nib.affines.apply_affine(imgs[0].affine, ijk))
        print(f"cut_coords not given; using peak |t| of '{pairs[0][0]}': {tuple(round(c) for c in cut)}")
    else:
        cut = tuple(args.cut_coords)

    vmax = args.vmax
    if vmax is None:
        vmax = max(float(np.nanmax(np.abs(im.get_fdata()))) for im in imgs)
    print(f"shared cut_coords={tuple(round(c) for c in cut)}, threshold={args.threshold}, vmax={vmax:.2f}")

    for (label, _), img in zip(pairs, imgs):
        title = f"{args.title + ', ' if args.title else ''}{label}"
        plotting.plot_stat_map(img, cut_coords=cut, threshold=args.threshold, vmax=vmax,
                               title=title, output_file=str(out / f"{label}_fixedview.png"))

    fig, axes = plt.subplots(len(pairs), 1, figsize=(9, 3.3 * len(pairs)))
    axes = np.atleast_1d(axes)
    for ax, (label, _), img in zip(axes, pairs, imgs):
        title = f"{args.title + ', ' if args.title else ''}{label}"
        plotting.plot_stat_map(img, cut_coords=cut, threshold=args.threshold, vmax=vmax, title=title, axes=ax)
    fig.tight_layout()
    combo = (args.title or "comparison").replace(" ", "_").replace(">", "_gt_")
    fig.savefig(out / f"{combo}_combined.png", dpi=150)
    plt.close(fig)
    print(f"\nwrote {out}/{combo}_combined.png and one PNG per label ({out}/<label>_fixedview.png)")


if __name__ == "__main__":
    main()

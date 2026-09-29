"""C1.5: test-retest reliability of GLM t-maps across two sessions (MNI-space maps written by script 15).

For each noise model and contrast, compares the session-'test' and session-'retest' t-maps inside the intersection of their brain masks:
  - spatial Pearson correlation of the unthresholded t-maps;
  - Dice overlap of voxels with t > --thr (default 3.1), and of the top 1% and top 5% of voxels (threshold-free; chance Dice = the fraction);
  - distance (mm) between the two sessions' peak voxels.
Sessions were preprocessed independently, so misregistration contributes to unreliability. One subject, one pair of runs: descriptive only.
Expected inputs: results/glm_bids/{mode}_{ses}_{noise}[_acN]_{task}_{contrast}_tmap.nii.gz and ..._mask.nii.gz from
`15_glm_on_bids.py` run for both sessions.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

CONTRASTS = {"fingerfootlips": ["Finger>others", "Foot>others", "Lips>others", "AllTask>rest"],
             "covertverbgeneration": ["Task>rest"]}


def tag(mode, ses, noise, ac, task, cname):
    return f"{mode}_{ses}_{noise}{'_ac' + str(ac) if ac else ''}_{task}_{cname.replace('>', '_gt_')}"


def dice(a, b):
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else float("nan")


def topk(x, frac):
    k = max(1, int(frac * x.size))
    m = np.zeros(x.size, bool)
    m[np.argpartition(-x, k - 1)[:k]] = True
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/glm_bids")
    ap.add_argument("--mode", default="fmriprep")
    ap.add_argument("--noise", nargs="+", default=["ols", "ar1_bc"])
    ap.add_argument("--acompcor", type=int, default=0)
    ap.add_argument("--thr", type=float, default=3.1)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    d = Path(args.dir)
    rows = []
    for noise in args.noise:
        for task, cons in CONTRASTS.items():
            for c in cons:
                t1n, t2n = (tag(args.mode, s, noise, args.acompcor, task, c) for s in ("test", "retest"))
                try:
                    a, ma, b, mb = (nib.load(d / f"{n}_{k}.nii.gz") for n in (t1n, t2n) for k in ("tmap", "mask"))
                except FileNotFoundError as e:
                    print(f"[skip] {noise} {task} {c}: missing files for {t1n} or {t2n}"); continue
                if a.shape != b.shape or not np.allclose(a.affine, b.affine):
                    print(f"[skip] {noise} {task} {c}: maps are on different grids {a.shape} vs {b.shape}"); continue
                inter = (ma.get_fdata() > 0) & (mb.get_fdata() > 0)
                x, y = a.get_fdata()[inter], b.get_fdata()[inter]
                ijk = np.argwhere(inter)
                p1, p2 = (nib.affines.apply_affine(a.affine, ijk[np.argmax(v)]) for v in (x, y))
                rows.append(dict(noise=noise, task=task, contrast=c, n_voxels=int(inter.sum()), r_t=float(np.corrcoef(x, y)[0, 1]),
                                 n_test_gt_thr=int((x > args.thr).sum()), n_retest_gt_thr=int((y > args.thr).sum()),
                                 dice_gt_thr=dice(x > args.thr, y > args.thr), dice_top1pct=dice(topk(x, 0.01), topk(y, 0.01)),
                                 dice_top5pct=dice(topk(x, 0.05), topk(y, 0.05)),
                                 peak_distance_mm=float(np.linalg.norm(p1 - p2))))
    if not rows:
        raise SystemExit("No map pairs found. Run 15_glm_on_bids.py for both sessions first (see the file header).")
    df = pd.DataFrame(rows)
    df.to_csv(Path(args.out) / "test_retest.csv", index=False)
    pd.set_option("display.width", 220)
    print(df.round(3).to_string(index=False))
    print("\nChance Dice equals the selected fraction (0.01 and 0.05). Sessions were preprocessed independently.")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels = [f"{r.task[:3]}:{r.contrast}" for r in df[df["noise"] == df["noise"].iloc[0]].itertuples()]
    for ax, col, title in ((axes[0], "r_t", "spatial correlation of t-maps"), (axes[1], "dice_top5pct", "Dice, top 5% of voxels")):
        for i, (noise, g) in enumerate(df.groupby("noise")):
            ax.bar(np.arange(len(g)) + i * 0.38, g[col], 0.38, label=noise)
        ax.set_xticks(np.arange(len(labels)) + 0.19); ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        ax.set_title(title); ax.legend()
    fig.tight_layout(); fig.savefig(Path(args.out) / "test_retest.png", dpi=150)


if __name__ == "__main__":
    main()

"""Sanity check for 17_test_retest.py's metric before trusting a low observed reliability number.

Loads one session's t-map (default: test, Finger>others) and computes r_t / Dice against:
  (a) itself                        -> should give r = 1.0, Dice = 1.0 (confirms the metric code is not broken)
  (b) a randomly permuted version    -> should give r ~ 0, Dice ~ chance = selected fraction (confirms the low
                                         observed numbers are not simply what ANY two unrelated maps would give)
If the real test-vs-retest numbers from 17_test_retest.py sit clearly between these two anchors (much lower than
(a), and only somewhat above (b) for the untresholded correlation), that supports reading them as real signal but
weak reliability, not as a broken comparison.
"""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def dice(a, b):
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else float("nan")


def topk(x, frac):
    k = max(1, int(frac * x.size))
    m = np.zeros(x.size, bool)
    m[np.argpartition(-x, k - 1)[:k]] = True
    return m


ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="results/glm_bids")
ap.add_argument("--tag", default="fmriprep_test_ar1_bc_fingerfootlips_Finger_gt_others")
ap.add_argument("--thr", type=float, default=3.1)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

t_img = nib.load(Path(args.dir) / f"{args.tag}_tmap.nii.gz")
m_img = nib.load(Path(args.dir) / f"{args.tag}_mask.nii.gz")
mask = m_img.get_fdata() > 0
x = t_img.get_fdata()[mask]
rng = np.random.default_rng(args.seed)
y_shuf = rng.permutation(x)

for name, y in (("self (should be ~1.0 / ~1.0)", x), ("randomly shuffled (should be ~0 / ~chance)", y_shuf)):
    r = float(np.corrcoef(x, y)[0, 1])
    d_thr = dice(x > args.thr, y > args.thr)
    d_top5 = dice(topk(x, 0.05), topk(y, 0.05))
    print(f"{name}: r={r:.3f}, Dice(t>{args.thr})={d_thr:.3f}, Dice(top5%)={d_top5:.3f} (chance top5%% = 0.05)")

print(f"\nCompare these anchors with the real test-vs-retest row for this contrast/model in results/test_retest.csv.")

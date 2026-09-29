"""Subject-specific functional ROI with split-half (odd/even blocks) selection.

Select the top-k% voxels on one half of the blocks, test on the other half, and
compare with the circular estimate (select and test on the same half).

Effects are reported as % of the subject's global mean intensity so subjects are
comparable. Under no information, the independent estimate should equal the
whole-brain mean, so `indep_minus_whole` is the quantity to look at. Chance
selection overlap (Dice) equals the selected fraction.

Limitation: halves come from the same run and are fit in one GLM, so they are
less independent than an across-run split (the demo data has one run per subject).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from fmriglm.data import get_runs, subject_masker
from fmriglm.pipeline import add_common_args, check_conditions, fit_run, run_contrast
from fmriglm.glm import t_stat

ap = add_common_args(argparse.ArgumentParser())
ap.add_argument("--top-frac", type=float, default=0.02, help="fraction of voxels in the fROI")
args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)


def split_events(ev, pos, neg):
    ev = ev[ev["trial_type"].isin([pos, neg])].sort_values("onset").reset_index(drop=True)
    for cond in (pos, neg):
        idx = ev.index[ev["trial_type"] == cond]
        ev.loc[idx[::2], "trial_type"] = cond + "_a"
        ev.loc[idx[1::2], "trial_type"] = cond + "_b"
    return ev


rows = []
for run in get_runs(args.data_dir):
    check_conditions(run.events, [args.pos, args.neg])
    ev = split_events(run.events, args.pos, args.neg)
    Y = subject_masker(run, args.fwhm).transform(run.img)
    g = float(Y.mean())                                   # global mean intensity
    conds = [args.pos + "_a", args.pos + "_b", args.neg + "_a", args.neg + "_b"]
    X, res = fit_run(Y, run, conds, args.hrf, args.drift, args.noise, events=ev)
    eA, vA = run_contrast(res, X, {conds[0]: 1.0, conds[2]: -1.0})
    eB, vB = run_contrast(res, X, {conds[1]: 1.0, conds[3]: -1.0})
    tA, tB = t_stat(eA, vA), t_stat(eB, vB)
    eA, eB = 100 * eA / g, 100 * eB / g                   # % of global mean intensity
    V = Y.shape[1]
    k = max(1, int(args.top_frac * V))
    selA, selB = np.argsort(tA)[-k:], np.argsort(tB)[-k:]
    indep = (eB[selA].mean() + eA[selB].mean()) / 2
    circ = (eA[selA].mean() + eB[selB].mean()) / 2
    whole = (eA.mean() + eB.mean()) / 2
    rows.append(dict(subject=run.subject, k_voxels=k, independent=indep, circular=circ,
                     whole_brain=whole, indep_minus_whole=indep - whole,
                     dice_A_vs_B=len(np.intersect1d(selA, selB)) / k, dice_chance=k / V))
df = pd.DataFrame(rows)
df.to_csv(out / "split_half_froi.csv", index=False)
print(df.round(4).to_string(index=False))
d = df["indep_minus_whole"]
print(f"\nmean (% of global mean intensity): independent={df['independent'].mean():.4f}  "
      f"circular={df['circular'].mean():.4f}  whole-brain={df['whole_brain'].mean():.4f}")
print(f"double-dipping inflation (circular - independent): {(df['circular'] - df['independent']).mean():.4f}")
print(f"independent - whole-brain: mean={d.mean():.4f}, positive in {(d > 0).sum()}/{len(d)} subjects")
try:
    print(f"one-sided Wilcoxon (independent > whole-brain): p={wilcoxon(d, alternative='greater').pvalue:.4f}  (n={len(d)}, descriptive)")
except ValueError as e:
    print("Wilcoxon not computed:", e)
print(f"median Dice A-vs-B={df['dice_A_vs_B'].median():.3f} (chance ~{df['dice_chance'].mean():.3f})")

fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(df)); w = 0.27
ax.bar(x - w, df["circular"], w, label="circular (select=test)")
ax.bar(x, df["independent"], w, label="independent half")
ax.bar(x + w, df["whole_brain"], w, label="whole-brain mean")
ax.set_xticks(x); ax.set_xticklabels(df["subject"])
ax.set_ylabel(f"{args.pos} - {args.neg} effect (% of global mean)"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(out / "split_half_froi.png", dpi=150)

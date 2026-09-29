"""B2: does the calibration problem depend on design frequency and TR? Resting-state data + fake designs.

Resting-state fMRI contains no task, so a fake block / event design cannot correlate with real task-driven activity:
every rejection of the contrast A - B is a false positive. Designs:
  periodic blocks at several periods (A block, then B block half a period later; block length = 0.24 x period, as in the
  language localizer), each with random phase; and a random event-related design (1 s events, random labels/onsets).
Models: ols, ar1, ar1_bc. Reported: two-sided false-positive rate at p<.05 and p<.001 (nominal 0.05 / 0.001), mean over
subjects (and phases), by TR and design frequency.

--decimate 2 keeps every 2nd scan (pseudo TR = 2 x TR). This is an approximation (no anti-alias filtering; temporal
smoothing of the original data is not preserved), NOT a substitute for data acquired at another TR.

Data: nilearn fetch_adhd (small NIAK-preprocessed resting-state subset). Whether it was temporally filtered is unverified;
per-subject spectral fractions are printed so you can check. Confounds: columns starting with 'motion' from the confounds file.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.signal import detrend

from fmriglm.design import build_design, make_contrast
from fmriglm.glm import fit_glm, contrast, t_stat, lag1_autocorr
from fmriglm.stats import t_to_p


def periodic_events(P, phase, total, duty=0.24):
    dur = duty * P
    rows, k = [], 0
    while phase + k * P < total:
        for label, off in (("A", 0.0), ("B", P / 2)):
            o = phase + k * P + off
            if o < total:
                rows.append((o, min(dur, total - o), label))
        k += 1
    return pd.DataFrame(rows, columns=["onset", "duration", "trial_type"])


def random_event_design(rng, total, dur=1.0, min_gap=2.0):
    n = min(60, int(total // 8)) // 2 * 2
    free = total - n * dur - (n - 1) * min_gap
    if n < 4 or free <= 0:
        raise ValueError("scan too short for a random event design")
    cuts = np.sort(rng.uniform(0.0, free, n))
    onsets = cuts + np.arange(n) * (dur + min_gap)
    labels = rng.permutation(["A"] * (n // 2) + ["B"] * (n // 2))
    return pd.DataFrame({"onset": onsets, "duration": dur, "trial_type": labels})


def band_fractions(Y, tr):
    """Fraction of (voxel-summed) power below 0.01 Hz, in 0.01-0.1 Hz, and above 0.1 Hz after linear detrending."""
    Yd = detrend(Y, axis=0)
    P = (np.abs(np.fft.rfft(Yd, axis=0)) ** 2).sum(1)
    f = np.fft.rfftfreq(Y.shape[0], d=tr)
    tot = P[f > 0].sum()
    return P[(f > 0) & (f < 0.01)].sum() / tot, P[(f >= 0.01) & (f < 0.1)].sum() / tot, P[f >= 0.1].sum() / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=10)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--tr", type=float, default=None, help="override the TR read from the NIfTI header")
    ap.add_argument("--periods", type=float, nargs="+", default=[12, 20, 30, 45, 60, 90])
    ap.add_argument("--decimate", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--n-phases", type=int, default=4)
    ap.add_argument("--models", nargs="+", default=["ols", "ar1", "ar1_bc"])
    ap.add_argument("--no-confounds", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    from nilearn.datasets import fetch_adhd
    from nilearn.maskers import NiftiMasker
    data = fetch_adhd(n_subjects=args.subjects, data_dir=args.data_dir)

    rows = []
    for i, (func, confpath) in enumerate(zip(data.func, data.confounds)):
        sid = f"{i + 1:02d}"
        tr = args.tr or float(nib.load(func).header.get_zooms()[3])
        Y0 = NiftiMasker(mask_strategy="epi", standardize=False).fit(func).transform(func)
        conf_df, cols = None, None
        if not args.no_confounds:
            cdf = pd.read_table(confpath)
            cols = [c for c in cdf.columns if c.lower().startswith("motion")]
            if cols and len(cdf) == Y0.shape[0]:
                conf_df = cdf
            else:
                print(f"[warn] subject {sid}: confounds unusable (cols={cols}, rows={len(cdf)} vs scans={Y0.shape[0]}); skipping them")
        lo, mid, hi = band_fractions(Y0, tr)
        print(f"subject {sid}: TR={tr:.3f}s, {Y0.shape[0]} scans, {Y0.shape[1]} voxels, motion cols={cols if conf_df is not None else None}, "
              f"power <0.01Hz/{'0.01-0.1'}/>0.1Hz = {lo:.2f}/{mid:.2f}/{hi:.2f}, "
              f"median voxel lag-1 autocorr = {np.median(lag1_autocorr(detrend(Y0, axis=0))):.2f}")
        for k in args.decimate:
            Y = Y0[::k]
            tr_eff = tr * k
            T = Y.shape[0]
            total = T * tr_eff
            conf = None if conf_df is None else conf_df.iloc[::k].reset_index(drop=True)
            designs = []
            for P in args.periods:
                if P < 6 * tr_eff or total < 3 * P:
                    continue
                for _ in range(args.n_phases):
                    designs.append(("periodic", P, periodic_events(P, rng.uniform(0, P), total)))
            for _ in range(args.n_phases):
                designs.append(("random_events", np.nan, random_event_design(rng, total)))
            for name, P, ev in designs:
                X = build_design(ev, ["A", "B"], T, tr_eff, "spm", "cosine", confounds=conf,
                                 confound_cols=cols if conf is not None else None)
                c = make_contrast(list(X.columns), {"A": 1.0, "B": -1.0})
                for model in args.models:
                    res = fit_glm(Y, X.to_numpy(), model)
                    p = t_to_p(t_stat(*contrast(res, c)), res.dof, "two-sided")
                    rows.append(dict(subject=sid, tr_eff=round(tr_eff, 3), design=name, period=P, model=model,
                                     fpr_p05=float((p < 0.05).mean()), fpr_p001=float((p < 0.001).mean())))
        print(f"  subject {sid} done ({len(designs)} designs per TR setting)")

    df = pd.DataFrame(rows)
    df.to_csv(out / "rest_null_calibration.csv", index=False)
    per_sub = df.groupby(["tr_eff", "design", "period", "model", "subject"], dropna=False)[["fpr_p05", "fpr_p001"]].mean().reset_index()
    summ = per_sub.groupby(["tr_eff", "design", "period", "model"], dropna=False)[["fpr_p05", "fpr_p001"]].agg(["mean", "std"])
    pd.set_option("display.width", 200)
    print("\nFalse-positive rates (mean and SD over subjects; nominal p05 = 0.05, p001 = 0.001):\n", summ.round(4).to_string())

    trs = sorted(per_sub["tr_eff"].unique())
    fig, axes = plt.subplots(2, len(trs), figsize=(5 * len(trs), 7), squeeze=False)
    for j, tr_eff in enumerate(trs):
        sub = per_sub[per_sub["tr_eff"] == tr_eff]
        periods = sorted(sub.loc[sub["design"] == "periodic", "period"].unique())
        for i, (col, nominal) in enumerate((("fpr_p05", 0.05), ("fpr_p001", 0.001))):
            ax = axes[i][j]
            for model in args.models:
                m = sub[sub["model"] == model]
                ys = [m[(m["design"] == "periodic") & (m["period"] == P)][col].mean() for P in periods]
                ys.append(m[m["design"] == "random_events"][col].mean())
                ax.plot(range(len(ys)), ys, marker="o", label=model)
            ax.axhline(nominal, color="k", ls="--", lw=0.8)
            ax.set_xticks(range(len(periods) + 1)); ax.set_xticklabels([f"{int(P)}s" for P in periods] + ["events"], fontsize=8)
            ax.set_xlabel("design period (blocks) / random event-related"); ax.set_ylabel(col)
            ax.set_title(f"TR = {tr_eff} s", fontsize=9)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "rest_null_calibration.png", dpi=150)


if __name__ == "__main__":
    main()

"""Fetch the demo dataset and print what is in it (run this first)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
from nilearn.image import load_img
from fmriglm.data import fetch_dataset, load_bids, collect_runs

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default=None)
args = ap.parse_args()

root = fetch_dataset(args.data_dir)
print("dataset root:", root)
runs = collect_runs(*load_bids(root))
subs = sorted({r.subject for r in runs})
print(f"{len(subs)} subjects, {len(runs)} runs: {subs}")
r = runs[0]
print("\nTR:", r.tr, "| slice_time_ref:", r.slice_ref)
print("image shape:", load_img(r.img).shape)
print("\nevents head:\n", r.events.head(12))
print("\ntrial_type counts:\n", r.events["trial_type"].value_counts())
print("\nduration by trial_type:\n", r.events.groupby("trial_type")["duration"].describe())
print("\nconfound columns:", None if r.confounds is None else list(r.confounds.columns))

"""Inspect a RAW BIDS directory (needs only nibabel + pandas): what is really there before preprocessing.

For each subject/session it lists T1w and BOLD files (flagging files that are not downloaded, e.g. git-annex/datalad
pointers), and prints image shape, voxel size and TR (from the NIfTI header), the JSON sidecar fields that matter for
preprocessing (RepetitionTime, SliceTiming present?, PhaseEncodingDirection, EchoTime), and the events file
(trial types, onset range, durations) compared with the scan duration. BIDS inheritance is followed: file-level
sidecar/events first, then the dataset-level task-<name>_bold.json / task-<name>_events.tsv.

Usage: python3 scripts/14_inspect_raw_bids.py /path/to/ds000114 [--sub 01]
"""
import argparse
import json
import re
from pathlib import Path

import nibabel as nib
import pandas as pd


def stem(p):
    return re.sub(r"\.nii(\.gz)?$", "", str(p))


def find_sidecar(f, root, task, suffix, ext):
    for cand in (Path(stem(f).rsplit("_bold", 1)[0] + f"_{suffix}.{ext}"),
                 root / f"task-{task}_{suffix}.{ext}"):
        if cand.exists():
            return cand
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bids_dir")
    ap.add_argument("--sub", default=None, help="only this subject label, e.g. 01")
    args = ap.parse_args()
    root = Path(args.bids_dir)

    dd = root / "dataset_description.json"
    if dd.exists():
        d = json.loads(dd.read_text())
        print(f"dataset: {d.get('Name')} | BIDSVersion {d.get('BIDSVersion')} | License {d.get('License')}")
    else:
        print("[warn] dataset_description.json not found")

    for sub in sorted(root.glob("sub-*")):
        if args.sub and sub.name != f"sub-{args.sub}":
            continue
        sessions = sorted(sub.glob("ses-*")) or [sub]
        for ses in sessions:
            print(f"\n=== {ses.relative_to(root)} ===")
            for f in sorted(ses.glob("anat/*.nii*")):
                if not f.exists():
                    print(f"  anat {f.name}: [NOT DOWNLOADED]"); continue
                h = nib.load(str(f)).header
                print(f"  anat {f.name}: shape {h.get_data_shape()}, voxel {tuple(round(float(z), 2) for z in h.get_zooms()[:3])} mm")
            for f in sorted(ses.glob("func/*_bold.nii*")):
                m = re.search(r"task-([A-Za-z0-9]+)", f.name)
                task = m.group(1) if m else "unknown"
                if not f.exists():
                    print(f"  func {f.name}: [NOT DOWNLOADED]"); continue
                h = nib.load(str(f)).header
                shp, zm = h.get_data_shape(), h.get_zooms()
                T = shp[3] if len(shp) > 3 else 1
                tr_hdr = float(zm[3]) if len(zm) > 3 else float("nan")
                print(f"  func {f.name}: shape {shp}, voxel {tuple(round(float(z), 2) for z in zm[:3])} mm, TR(header) {tr_hdr:.3f} s")
                sc = find_sidecar(f, root, task, "bold", "json")
                if sc is not None:
                    j = json.loads(sc.read_text())
                    st = j.get("SliceTiming")
                    print(f"       sidecar {sc.relative_to(root)}: RepetitionTime={j.get('RepetitionTime')}, "
                          f"SliceTiming={'present (%d values)' % len(st) if st else 'ABSENT'}, "
                          f"PhaseEncodingDirection={j.get('PhaseEncodingDirection')}, EchoTime={j.get('EchoTime')}")
                else:
                    print("       sidecar JSON: NOT FOUND (fMRIPrep needs RepetitionTime; slice timing correction needs SliceTiming)")
                ev = find_sidecar(f, root, task, "events", "tsv")
                if ev is not None:
                    e = pd.read_csv(ev, sep="\t")
                    dur = T * (j.get("RepetitionTime", tr_hdr) if sc is not None else tr_hdr)
                    end = float((e["onset"] + e["duration"]).max()) if {"onset", "duration"} <= set(e.columns) else float("nan")
                    print(f"       events {ev.relative_to(root)}: {len(e)} rows; scan duration {dur:.1f} s; last event ends {end:.1f} s")
                    if "trial_type" in e.columns:
                        print("       trial_type counts:", e["trial_type"].value_counts().to_dict())
                        print("       mean duration by trial_type:", e.groupby("trial_type")["duration"].mean().round(2).to_dict())
                else:
                    print("       events: NOT FOUND")


if __name__ == "__main__":
    main()

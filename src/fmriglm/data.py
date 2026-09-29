"""Dataset access (nilearn language localizer demo dataset, BIDS)."""
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

TASK = "languagelocalizer"


@dataclass
class Run:
    subject: str
    img: Any                       # path or Nifti image (preprocessed BOLD)
    events: pd.DataFrame
    confounds: Optional[pd.DataFrame]
    tr: float
    slice_ref: float


def fetch_dataset(data_dir=None):
    """Download (or reuse) the demo dataset; returns its root directory."""
    from nilearn.datasets import fetch_language_localizer_demo_dataset as fetch
    try:
        return fetch(data_dir=data_dir, legacy_output=False).data_dir
    except TypeError:                       # older nilearn without legacy_output
        out = fetch(data_dir=data_dir)
        return out[0] if isinstance(out, (tuple, list)) else out.data_dir


def load_bids(data_dir, task=TASK):
    """Use nilearn only to *discover* files; all modelling is ours."""
    from nilearn.glm.first_level import first_level_from_bids
    kw = dict(img_filters=[("desc", "preproc")])
    try:
        return first_level_from_bids(data_dir, task, space_label="", **kw)
    except Exception as e:                  # signature differs across nilearn versions
        print(f"[warn] first_level_from_bids with space_label='' failed ({e!r}); retrying without")
        return first_level_from_bids(data_dir, task, **kw)


def collect_runs(models, imgs, events, confounds):
    runs = []
    for m, im_list, ev_list, cf_list in zip(models, imgs, events, confounds):
        if not isinstance(im_list, (list, tuple)):
            im_list, ev_list = [im_list], [ev_list]
            cf_list = [cf_list]
        if cf_list is None:
            cf_list = [None] * len(im_list)
        for im, ev, cf in zip(im_list, ev_list, cf_list):
            runs.append(Run(subject=str(m.subject_label), img=im, events=ev, confounds=cf,
                            tr=float(m.t_r), slice_ref=float(getattr(m, "slice_time_ref", 0.0) or 0.0)))
    return runs


def get_runs(data_dir=None, task=TASK):
    root = fetch_dataset(data_dir)
    return collect_runs(*load_bids(root, task))


def subject_masker(run, fwhm=None):
    from nilearn.maskers import NiftiMasker
    return NiftiMasker(mask_strategy="epi", smoothing_fwhm=fwhm, standardize=False).fit(run.img)

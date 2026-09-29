# fmri-language-glm

A methods-focused study of the fMRI General Linear Model on a language-localizer contrast
(reading language vs. a matched control). The GLM, design construction, AR(1) prewhitening
and group-level inference are implemented **from scratch in NumPy**, validated against nilearn and against
simulations with known truth. The point is *how modelling and inference choices change the answer*, including
where the standard pipeline is not well calibrated.

**Data:** nilearn's language localizer demo dataset: 10 subjects, 1 run each, TR 1.5 s, 229 scans, 45×53×43
voxels, preprocessed and normalised to MNI space, conditions `language` and `string` (12 blocks of ~6.8 s each,
strictly periodic, period ≈ 28.6 s). It is downloaded at runtime and never committed. **No preprocessing is implemented here; section 11 adds one subject of a raw dataset that was preprocessed by running fMRIPrep (run and audited, not written by me).**

## Summary of findings

1. The from-scratch GLM matches nilearn (Pearson r = 0.99991, subject 01); with white noise, OLS is calibrated
   and contrast estimates are unbiased (relative bias within ±2%).
2. The HRF model matters far more than drift or noise-model choice.
3. First-level p-values are **anticonservative** on real data under two different null designs in every subject.
   Per-voxel AR(1) prewhitening, as in nilearn's default, has its own small anticonservative bias (shown on
   synthetic white noise in both nilearn and this code). A bias-corrected AR(1) (`ar1_bc`; a known correction from
   Worsley et al. 2002, re-implemented here) helps in all 10 subjects but does not fully fix the inflation.
4. Phase-randomised surrogates of each subject's own noise reproduce the real inflation in level and in
   subject ordering, and AR(1) barely helps on them: consistent with inflation coming from noise-spectrum
   structure near the task frequency rather than lag-1 autocorrelation (exploratory; see caveats).
5. Motion is mild and does not explain the between-subject differences; motion-regressor choice does not fix it.
6. Group-level voxelwise evidence is weak (10 subjects); a pre-specified atlas-ROI analysis shows a left-lateralised
   fronto-temporal pattern, but the effect is not perfectly specific (a control region shows a left>right asymmetry).
7. Split-half fROIs show clear double-dipping inflation (circular ≈ 2–4× independent).
8. On unrelated resting-state data with fake designs, false-positive inflation grows with block period (near nominal for 12 s blocks, clearly
   inflated for 60–90 s blocks) and is smaller for random event-related designs; the bias-corrected AR(1) is best or tied almost everywhere but does
   not fix slow designs. Plain per-voxel AR(1) becomes worse than OLS when scans are very short, as the bias explanation predicts.
9. On one raw subject preprocessed with fMRIPrep (ds000114), a finger-foot-lips analysis with `ar1_bc` is close to calibrated under a surrogate-noise null while OLS is clearly
   inflated; a single-condition covert-verb-generation vs rest contrast (7 blocks, 60 s period) is inflated even under the bias-corrected model and has a large excess beyond
   noise, so it is not interpretable as a localised map.

## Results (single execution on the demo dataset; raw outputs in `results/`, git-ignored)

### 1. Validation vs nilearn (`02`, subject 01, spm HRF, cosine drift, AR(1))
dof = 214; 187 voxels pass FDR q<.05; max t = 8.83; r = 0.99991, mean |Δt| = 0.013 vs nilearn.
Note: nilearn shares the modelling strategy, so agreement shows equivalence, not correctness; see section 5.

### 2. Design ablation (`03`, subject 01; descriptive only)
Boxcar models give 0–4 FDR voxels, canonical HRF models 165–271 (max t ≈ 8–11 vs ≈ 4.4). Drift choice changes counts by
tens of voxels. AR(1) lowers median-absolute residual lag-1 autocorrelation from ≈0.09–0.11 to ≈0.01. More significant voxels
is not better; see section 3.

### 3. Null calibration (two-sided; nominal p<.05 → 0.05, p<.001 → 0.001)
- **Random block designs** (`04`, subjects 01–06, per-subject medians): p<.05: OLS 0.062–0.092, AR(1) 0.064–0.075.
- **Quarter-period circular shifts** (`07`, 10 subjects; measured |corr| with the true contrast ≤ 0.06). Mean of per-subject
  medians:

| model | p<.05 | p<.001 |
|---|---|---|
| OLS | 0.0934 | 0.0091 |
| AR(1) | 0.0892 | 0.0056 |
| AR(1), bias-corrected (`ar1_bc`) | 0.0747 | 0.0037 |

  `ar1_bc` is lower than both OLS and AR(1) in all 10 subjects. Inflation stays large in some (subject 02: 0.110 at p<.05,
  0.0086 at p<.001 with `ar1_bc`).

### 4. Known-truth simulation (`11`; effect of 0.5 noise SD injected in 20% of voxels; task-responsive voxels excluded)
FPR at p<.05 (mean over subjects; nominal 0.05):

| noise | OLS | AR(1) | `ar1_bc` |
|---|---|---|---|
| white | 0.0500 | 0.0630 | 0.0517 |
| synthetic AR(1), rho 0.4 | 0.1818 | 0.0810 | 0.0636 |
| surrogate_full (task fitted out) | 0.0322 | 0.0330 | 0.0267 |
| surrogate_nuis (nuisance-only residuals) | 0.0856 | 0.0882 | 0.0735 |

- Relative bias of the contrast estimate ≈ 0 (within ±1–2%) for all noise types.
- `surrogate_full` is conservative by construction (fitting the task regressors removes noise power near the task
  frequency). `surrogate_nuis` keeps the real spectrum and matches the real quarter-shift rates: mean 0.0856 vs 0.0934 (OLS) and
  0.0882 vs 0.0892 (AR(1)); Spearman across subjects +0.94 (OLS) and +0.98 (AR(1)) (n = 10, exploratory).
- Power at p<.001 for this effect size is about 21% under white noise with a calibrated test (one 6-minute run). Power values for
  miscalibrated models are not comparable.

### 5. Reference-implementation check on synthetic data (`12`; ~40,000 voxels per case; nilearn 0.14.1)
White noise, FPR at p<.05: nilearn OLS 0.0489, ours 0.0491; nilearn AR(1) 0.0585, ours 0.0594; ours `ar1_bc` 0.0500.
AR(1) noise (rho 0.4): nilearn OLS 0.1868, ours 0.1870; nilearn AR(1) 0.0735, ours 0.0726; ours `ar1_bc` 0.0580.
The inflation of per-voxel AR(1) is shared by the reference implementation. This design has 9 regressors (no confounds), so the
bias is smaller than in the real 15-regressor analysis.

### 6. QC (`08`) and motion sensitivity (`09`)
Mean framewise displacement 0.08–0.20 mm; only subject 07 has many scans above 0.5 mm (8; max FD 1.76 mm). Rotation units were
auto-detected as radians (pooled max 0.18; not independently confirmed). tSNR ranges 38.8–89.1 (subjects 01 and 02 are much higher
than the rest). Median lag-1 residual autocorrelation is near zero (−0.04 to +0.02). Spearman correlations of QC metrics with
per-subject inflation (10 correlations, uncorrected, n = 10, exploratory): mean FD +0.24 (OLS), +0.39 (AR(1)); the residual
power near the task frequency relative to high frequencies correlates at +0.66 (p = 0.038) for OLS and +0.53 (p = 0.117) for AR(1),
which does not survive correction for 10 tests.
Motion-regressor sets (none / 6 / 12 / 24 parameters / scrubbing) change mean false-positive rates by about 0.01 or less and do not reduce the
inflation (AR(1) at p<.05: 0.083 with no motion regressors → 0.089 with 6 → 0.094 with 24). The median voxel count passing FDR in the true design is
110–142 across variants, with large between-subject spread (3 to 1,234 voxels).

### 7. Group inference (`05`, 10 subjects, dof = 9; sign-flip with all 1024 patterns enumerated)

| FWHM | uncorrected p<.001 | FDR q<.05 | Bonferroni | sign-flip FWE | cluster p<.001 & ≥10 vox (heuristic) |
|---|---|---|---|---|---|
| 0 mm | 32 | 2 | 1 | 2 | 0 |
| 4 mm | 41 | 2 | 1 | 2 | 0 |
| 6 mm | 116 | 6 | 1 | 2 | 57 |
| 8 mm | 196 | 83 | 1 | 4 | 99 |

Four smoothing levels were examined; all are reported. Sign-flip FWE is the primary inference. At 6 mm, raw-unit and %-of-global-mean effect
maps give identical counts (FDR 6, FWE 2).

### 8. Anatomical validation (`10`, Harvard-Oxford atlas; analysis pre-specified in the script header; 6 mm, AR(1))
- The 6 FDR voxels are single voxels (not clusters). The two that pass FWE are in the **right** hemisphere (temporal pole, MNI ≈ (55, 10, −14), t = 10.1;
  posterior supramarginal gyrus, ≈ (55, −44, 14), t = 9.0); their left counterparts pass FDR only. The temporal pole is a
  signal-dropout-prone region, so a single-voxel peak there needs a local QC check.
- Laterality: group LI 0.48–0.61 over cortex and 0.51–0.73 in language ROIs (t>3, 4, 5); per-subject LI median 0.60, 9/10 left-lateralised (subject 10: −0.05).
- ROIs (effect as % of global mean intensity; 27 rows, BH-FDR): left STG anterior 0.42, left STG posterior 0.35, left MTG anterior 0.35, left IFG
  pars opercularis 0.13 are positive in 10/10 subjects and FDR-significant; left>right differences of 0.18–0.23 in those regions.
  Left IFG pars triangularis, left posterior MTG and angular gyrus are not significant; right angular gyrus is significantly negative.
  Control ROIs: occipital pole is negative (not FDR-significant); **precentral gyrus shows a significant left>right difference (0.082, 10/10)**, so laterality is
  not perfectly language-specific (language-ROI asymmetries are 2–3× larger).

### 9. Split-half functional ROI (`06`; top 2% of voxels; % of global mean intensity)

| FWHM | independent | circular | whole-brain | indep − whole (positive subjects) | median Dice (chance 0.02) |
|---|---|---|---|---|---|
| 0 mm | 0.41 | 1.70 | −0.07 | +0.48 (9/10) | 0.11 |
| 6 mm | 0.43 | 0.91 | −0.06 | +0.49 (10/10) | 0.25 |

One-sided Wilcoxon p = 0.0049 and 0.0010 (the minimum attainable with n = 10 is 1/1024): descriptive only.

### 10. Resting-state null with fake designs (`13`; nilearn ADHD sample, 10 subjects, motion regressors included)
Acquisition groups (from the file headers): subjects 01–04, TR 2.0 s, 175–176 scans; subjects 05–06, TR 2.5 s, **77 scans**; subjects 07–10,
TR 1.96 s, 260–261 scans. After linear detrending, 20–62% of power lies above 0.1 Hz and 4–29% below 0.01 Hz, so there is no sign of strict band-pass
filtering (the temporal preprocessing of this subset was not verified independently). Median voxel lag-1 autocorrelation 0.07–0.40.
Decimating by 2 gives pseudo-TRs (3.92 s for subjects 07–10, ~130 scans; 4.00 s for 01–04, ~88 scans; 5.00 s for 05–06, ~39 scans); this is an
approximation (no anti-alias filtering; fewer scans), not data acquired at another TR. Every rejection is a false positive.
Two-sided FPR at p<.05, mean over subjects (nominal 0.05), shown as OLS / AR(1) / `ar1_bc`:

| setting (n subjects, scans) | 30 s blocks | 60 s blocks | random event-related |
|---|---|---|---|
| TR 1.96 (4, ~260) | 0.087 / 0.077 / 0.061 | 0.112 / 0.099 / 0.079 | 0.072 / 0.063 / 0.051 |
| TR 2.00 (4, ~176) | 0.102 / 0.094 / 0.073 | 0.100 / 0.090 / 0.069 | 0.073 / 0.068 / 0.054 |
| TR 2.50 (2, 77) | 0.148 / 0.115 / 0.083 | 0.145 / 0.110 / 0.075 | 0.105 / 0.085 / 0.067 |
| pseudo-TR 3.92 (4, ~130) | 0.058 / 0.073 / 0.053 | 0.071 / 0.090 / 0.062 | 0.060 / 0.070 / 0.057 |
| pseudo-TR 4.00 (4, ~88) | 0.069 / 0.089 / 0.065 | 0.069 / 0.095 / 0.062 | 0.055 / 0.069 / 0.055 |
| pseudo-TR 5.00 (2, ~39) | 0.065 / 0.093 / 0.060 | 0.072 / 0.108 / 0.066 | 0.051 / 0.068 / 0.054 |

Also: within the two four-subject native groups, OLS at p<.05 rises with block period (TR 1.96: 0.058, 0.074, 0.087, 0.093, 0.112, 0.121 for periods
12, 20, 30, 45, 60, 90 s); at p<.001 it reaches 0.014 for 90 s blocks (14× nominal). SDs across subjects are large at 30 s (0.036–0.064).
Notes: the TR 2.5 group has only two short scans, so its larger inflation cannot be attributed to TR (it is confounded with scan length and n = 2);
the same two subjects, decimated, show AR(1) worse than OLS, as expected when the residual-based rho is biased by about −p/T with T small, and
`ar1_bc` corrects it. For the same subjects, coarser sampling lowers OLS inflation (60 s blocks: 0.112 → 0.071 for TR 1.96 → 3.92; 0.100 → 0.069 for
TR 2.00 → 4.00), directionally consistent with the literature on short TRs. Near a 30 s period and TR ≈ 2 s, unrelated rest data give OLS 0.087–0.102, comparable
to the language localizer (period 28.6 s, TR 1.5 s; OLS 0.093), consistent with design frequency and noise, not the task, driving the inflation.
This dataset does not cover short TRs (< 1 s, multiband).

### 11. One raw subject preprocessed with fMRIPrep (`14`, `15`, `16`; OpenNeuro ds000114, CC0)
Subject 01, session `test`. Tasks: finger-foot-lips (184 volumes; Finger, Foot, Lips, 5 blocks of 15 s each) and covert verb generation (173 volumes; ONE condition, 7 blocks of
30 s, against implicit rest, so it is task-vs-rest, not a language-vs-control contrast). TR 2.5 s; 4 mm functional voxels; T1w 1.0 × 1.3 × 1.0 mm; slice timing present in the
sidecars; no fieldmap and no PhaseEncodingDirection, so **no susceptibility-distortion correction** was possible. The first event is at 10 s (four volumes), consistent with dummy scans
still being in the files (inferred). fMRIPrep 25.2.5 was run with `fmriprep-docker` on an Apple-Silicon Mac (amd64 emulation; about one hour of wall-clock time by log timestamps) with
`--fs-no-reconall` and output space MNI152NLin2009cAsym at 2 mm; it flagged one non-steady-state volume. The GLM used 6 motion parameters, the non-steady-state column, canonical
HRF, cosine drift and `ar1_bc`.
Motion: mean FD 0.24 mm (finger-foot-lips; max 3.29 mm, 15 scans above 0.5 mm) and 0.23 mm (verb generation; max 2.86 mm, 7 scans); the largest |correlation| between any motion
parameter and the task regressor is 0.17.

**Visual QC (retest session; directly inspected, not just summarised):** T1w-to-MNI normalisation looks accurate (ventricle shape/size and brain outline match the template at every
slice shown, no clipping). BOLD-to-T1 coregistration contours generally follow tissue boundaries in the EPI reference at mid/superior slices for both tasks; inferior temporal/orbital
slices are noisy with visible signal dropout (expected without a fieldmap), making alignment harder to verify there specifically. The brain mask tightly follows the actual EPI brain
outline (no gross over/under-inclusion). The finger-foot-lips carpet plot shows one clear, identifiable event around 5:05-5:20 (of a 7:40 run): a vertical band crossing every tissue
compartment simultaneously, coincident with the largest FD spike (2.68 mm) and the DVARS maximum (13.5) - a single severe head-motion event, not a diffuse problem. The verb-generation
carpet plot has no comparably dominant single spike; instead cortical grey matter shows a more diffuse, loosely periodic banding pattern, which is at least qualitatively consistent
with widespread task-locked BOLD modulation given the 30s-on/30s-off design (a surrogate/phase-randomised null could not reproduce timing-locked structure) - offered as a qualitative
observation, not proof, alongside the quantitative surrogate-null result above. The test-session report was checked visually in-browser by the user (reported as normal) but not
independently re-inspected image-by-image here.

Raw (no preprocessing, native space) vs fMRIPrep (MNI, 2 mm): FDR-significant voxels / robust SD of t (voxel counts are not comparable: 17,210 vs 228,242 voxels in the masks):

| contrast | raw | fMRIPrep |
|---|---|---|
| Finger>others | 14 / 1.11 | 314 / 1.11 |
| Foot>others | 27 / 1.25 | 552 / 1.14 |
| Lips>others | 53 / 1.15 | 4,605 / 1.45 |
| AllTask>rest | 554 / 1.48 | 1,750 / 1.10 |
| Covert verb generation | 3,646 / 2.34 | 36,108 / 2.16 |

Peaks (fMRIPrep, Harvard-Oxford labels): Lips: bilateral postcentral gyrus, (54, −8, 34) and (−50, −14, 40); Finger: right precentral gyrus, (44, −16, 64) (which hand tapped is
not stated in the dataset); Foot: the strongest cluster (269 voxels) peaks at (12, −50, 80), at the very top of the volume and outside the atlas, which needs visual verification
(a medial foot area is expected, but this is more posterior and higher than usual; it may be vertex motor cortex or an edge effect). Verb generation: giant clusters (13,271 and 15,333 voxels)
peaking at the brain's outer boundary and top, e.g. (70, −44, 6) and (8, −68, 68). Raw-mode peaks are in native space and not anatomically labelled.

Surrogate-noise null (nuisance-only residuals of voxels that are not clearly task-responsive, phase-randomised, fitted with the real design; a lower bound on noise-driven width):

| task / contrast | model | observed robust SD | null robust SD | null FPR p<.05 | p<.001 |
|---|---|---|---|---|---|
| FFL AllTask>rest | `ar1_bc` / OLS | 1.10 / 1.33 | 1.06 / 1.31 | 0.068 / 0.140 | 0.0027 / 0.0161 |
| FFL Finger>others | `ar1_bc` / OLS | 1.11 / 1.53 | 1.02 / 1.30 | 0.057 / 0.138 | 0.0022 / 0.0191 |
| FFL Foot>others | `ar1_bc` / OLS | 1.14 / 1.35 | 0.98 / 1.23 | 0.048 / 0.119 | 0.0013 / 0.0132 |
| FFL Lips>others | `ar1_bc` / OLS | 1.46 / 1.76 | 1.02 / 1.26 | 0.056 / 0.125 | 0.0018 / 0.0130 |
| Verb generation | `ar1_bc` / OLS | 2.16 / 2.38 | 1.47 / 1.58 | 0.183 / 0.219 | 0.0282 / 0.0460 |
| Verb generation + 5 aCompCor | `ar1_bc` / OLS | 1.83 / 1.91 | 1.18 / 1.21 | 0.104 / 0.114 | 0.0087 / 0.0124 |

A sanity check (`18`) confirms the test-retest metric itself is sound: comparing one map against itself gives r=1.000, Dice=1.000 exactly; comparing it against a randomly permuted
version of itself gives r≈0.000 and Dice(top 5%)≈0.049, matching the 0.05 chance level. The real test-vs-retest numbers above (r=0.11-0.24, Dice-top5%=0.08-0.23) sit clearly between
these two anchors: weak but non-zero, real agreement - not a broken comparison.

For verb generation with `ar1_bc`, 11.6% of voxels have t > 3 (null 2.4%) and 6.2% have t > 4 (null 0.5%). Adding 5 aCompCor components lowers the FDR count from 36,108 to 19,868 and the
observed width from 2.16 to 1.83, but the components correlate with the task regressor by −0.15 to +0.43 (0.36, 0.27 and 0.43 for components 01, 03, 04), so this may remove task-related
variance and is reported as a sensitivity analysis, not a correction. The carpet plot for this run (section 11) shows diffuse, loosely task-periodic banding across grey matter rather
than a single motion event, which is at least qualitatively compatible with part of the excess being genuine widespread response rather than only artifact - this is a visual
impression, not independent statistical evidence.

### 12. Test-retest reliability across sessions (`17`; ds000114 sub-01, ses-test vs ses-retest, both preprocessed independently with fMRIPrep)
Both sessions' HTML QC reports report no pipeline errors. Retest-session figures were directly inspected (anatomical normalisation, both tasks' coregistration, carpet plots, brain
mask; see section 11): no gross registration failure, but one identifiable severe motion event in the finger-foot-lips run and expected inferior signal dropout from the missing
fieldmap. Test-session figures were checked only visually in-browser by the user. Neither check rules out subtle between-session registration differences (only gross failures), which
is the main unquantified factor that could inflate the low reliability below beyond what genuine neural/statistical variability would produce.
Spatial Pearson correlation of the two sessions' unthresholded t-maps, and Dice overlap of thresholded/top-voxel sets, computed over the intersection of the
two sessions' brain masks (~210,000 voxels; one subject, one session pair — descriptive only):

| contrast | model | r (t-map) | Dice, t>3.1 | Dice, top 1% | Dice, top 5% | peak distance (mm) |
|---|---|---|---|---|---|---|
| Finger>others | OLS / `ar1_bc` | 0.110 / 0.113 | 0.109 / 0.208 | 0.189 / 0.161 | 0.124 / 0.118 | 6.0 / 18.3 |
| Foot>others | OLS / `ar1_bc` | 0.133 / 0.128 | 0.106 / 0.109 | 0.107 / 0.105 | 0.117 / 0.116 | 28.4 / 26.3 |
| Lips>others | OLS / `ar1_bc` | 0.109 / 0.108 | 0.126 / 0.147 | 0.204 / 0.192 | 0.125 / 0.125 | 10.0 / 112.2 |
| AllTask>rest | OLS / `ar1_bc` | 0.225 / 0.232 | 0.200 / 0.209 | 0.210 / 0.208 | 0.228 / 0.230 | 57.6 / 118.8 |
| Verb generation, Task>rest | OLS / `ar1_bc` | 0.229 / 0.239 | 0.281 / 0.270 | 0.079 / 0.078 | 0.175 / 0.168 | 84.1 / 97.9 |

Voxel counts passing FDR also differ substantially between sessions for the same contrast and model (e.g. `ar1_bc` Finger>others: 314 test vs 4 retest; verb
generation `ar1_bc`: 36,108 test vs 52,047 retest).

**Reading it:** spatial correlations (0.11–0.24) and Dice overlaps (0.08–0.28, barely above the chance level of the selected fraction for the top-1%/5%
comparisons) indicate poor voxel-level reproducibility of single-session activation maps in this data. This is not an isolated or surprising result: it lands
inside the range reported by the task-fMRI test-retest literature. Bennett & Miller (2010) first raised the concern; Elliott et al. (2020, *Psychological
Science*), a meta-analysis of 90 studies plus two independent test-retest samples (Human Connectome Project, Dunedin Study), reported a mean ICC of 0.397
across common task-fMRI measures and *a priori* ROI-level ICCs ranging 0.067–0.485, concluding that "commonly used task-fMRI measures generally do not have
the test-retest reliability necessary for biomarker discovery or brain-behavior mapping." Kragel et al. (2021) offer a partial counterpoint (reliability
depends heavily on what is measured and how). The present numbers are broadly consistent with that literature's low end, which is expected given this
dataset's small block counts (5–7 per condition), lack of distortion correction, and independently-run registrations per session.
**Calibration is not reliability:** `ar1_bc` was shown (sections 4, 5, 11) to be closer to nominal false-positive rates within a session than OLS, but it is
NOT more reproducible across sessions here — spatial correlations and Dice are nearly identical between the two models, and peak locations for `ar1_bc` are
if anything less stable for two contrasts (Lips, AllTask). A model can be better-calibrated without being more reliable; these are different properties and
one does not imply the other.

## What I would and would not claim

- Would claim: correct OLS core verified against known truth and nilearn; HRF modelling dominates; standard first-level inference is
  miscalibrated on this data and per-voxel AR(1) is itself slightly anticonservative (in nilearn too); a known bias correction helps;
  subject-level inflation is reproduced by phase-randomised noise with each subject's spectrum; double-dipping is large; the ROI pattern is left-lateralised
  and only partly language-specific; on rest data with fake designs, inflation grows with block period and the bias-corrected AR(1) helps throughout.
- Would claim (continued): single-session voxelwise task-fMRI maps in a small, distortion-uncorrected dataset are not reliably reproducible across sessions, consistent with the
  published test-retest-reliability literature; better first-level calibration did not improve cross-session reliability in this data.
- Would not claim: a proven cause for the remaining inflation; a novel method; a full language-network map; any claim of having implemented preprocessing (fMRIPrep was run and its outputs audited; it was not written here);
  that the specific ICC/correlation values generalise beyond this one subject and dataset.

## Limitations

- Small, preprocessed demo dataset; one run per subject; 10 subjects; one TR and one block design. Group p-values are coarse.
- `surrogate_nuis` retains power near the task frequency, which may include weak sub-threshold task signal; the quarter-shift null could leak the same
  way. Both nulls are diagnostics, not proofs. Both use the same 10 subjects, so their agreement is not independent evidence.
- The atlas is coarse relative to the ~4 mm group grid; labels near boundaries are unreliable; hemisphere is defined by the sign of the MNI x coordinate.
- Split-half halves come from one run and one GLM. Effects are relative to global mean intensity, not local baseline.
- Resting-state check: only ten subjects from three acquisition settings (one of them two subjects with 77 scans), with unverified temporal preprocessing; decimation is an
  approximation to a longer TR; no short-TR (multiband) data. TR and scan length are confounded in places.
- fMRIPrep section: one subject; no distortion correction (no fieldmap or phase-encoding metadata); verb generation is task-vs-rest with only 7 blocks; the surrogate null excludes
  responsive voxels and is a lower bound; the slice-timing reference (0.5 TR) is an assumption. Retest-session QC figures were directly inspected (see above); test-session figures
  were checked only visually in-browser by the user, not re-inspected here. Reading a carpet plot's banding as "consistent with task signal" is a qualitative visual impression, not a
  statistical test.
- Test-retest: one subject, one session pair, independently preprocessed (so some unreliability reflects registration differences, not only neural/statistical noise); reliability
  metrics (spatial r, Dice) were not benchmarked against a self-correlation or shuffled-map control before being reported.
- The cluster-extent threshold is a heuristic, not FWE control. Several analysis variants were explored (smoothing levels, nulls, models, ROI sets); all are reported.

## Corrections made along the way (kept for transparency)

- Motion regressors were initially dropped silently (column names differ in this dataset); fixed.
- A circular-shift null was replaced because the design is strictly periodic; random-block and quarter-shift nulls were used instead.
- I first attributed the AR(1) shortfall to per-voxel estimation noise; a check showed a global rho did not help. The cause is a
  negative bias in the residual-based estimate, which a design-specific correction removes on white noise.
- The first surrogate experiment was conservative because task regressors had been fitted out of the residuals; corrected with nuisance-only residuals.
- I first read the TR 2.5 s resting-state group as a TR effect; the file headers showed it is two subjects with only 77 scans, so it is confounded with scan length.
- An early reading of group peaks as left-lateralised came from an image at a different smoothing level; the atlas analysis shows bilateral voxelwise peaks.

## Reproduce

```bash
pip install -r requirements.txt && pip install -e .
python3 -m pytest -q tests
python3 scripts/01_inspect_dataset.py
python3 scripts/02_first_level_from_scratch.py
python3 scripts/03_design_ablation.py
python3 scripts/04_null_calibration.py --n-null 100 --run-index 0     # repeat for other run indices
python3 scripts/07_null_quarter_shift.py                               # ols, ar1 -> null_quarter_shift.csv
python3 scripts/07_null_quarter_shift.py --models ols ar1 ar1_bc      # -> null_quarter_shift_models.csv
python3 scripts/08_qc_metrics.py
python3 scripts/09_motion_sensitivity.py
python3 scripts/05_group_analysis.py --out results/group_fwhm0
for f in 4 6 8; do python3 scripts/05_group_analysis.py --fwhm $f --out results/group_fwhm$f; done
python3 scripts/06_split_half_froi.py --out results/froi_fwhm0
python3 scripts/06_split_half_froi.py --fwhm 6 --out results/froi_fwhm6
python3 scripts/10_atlas_validation.py
python3 scripts/11_simulation_known_truth.py
python3 scripts/12_reference_check_synthetic.py
python3 scripts/13_rest_null_design_frequency.py --subjects 10   # downloads the nilearn ADHD sample
# raw dataset (download ds000114 from OpenNeuro into data/ds000114, build data/ds000114_subset with sub-01/ses-test anat + the two tasks), then:
python3 scripts/14_inspect_raw_bids.py data/ds000114 --sub 01
fmriprep-docker data/ds000114_subset data/derivatives participant --participant-label 01 --fs-license-file ~/license.txt --fs-no-reconall \
  --output-spaces MNI152NLin2009cAsym:res-2 --skip-bids-validation --nthreads 4 --mem_mb 10000 -w ~/fmriprep_work   # needs Docker and a FreeSurfer license
python3 scripts/15_glm_on_bids.py --mode raw --task fingerfootlips
python3 scripts/15_glm_on_bids.py --mode fmriprep --task fingerfootlips --label-peaks
python3 scripts/16_qc_and_surrogate_null_bids.py --task covertverbgeneration --acompcor 5
# repeat 15 for --ses retest against data/ds000114_retest + data/derivatives_retest, for both tasks and noise in {ols, ar1_bc}, then:
python3 scripts/17_test_retest.py
```

Common flags: `--pos --neg --hrf --drift --noise --fwhm --out --data-dir`. The Harvard-Oxford atlas is fetched through nilearn on first use (needs internet).

## Still to do (see `TODO.md` if present)

Empirical-null check on the true-design t-maps; higher-order AR / spectrum-aware whitening; a permutation cluster-FWE test; short-TR (multiband) generalisation with another rest dataset; a remedy test for slow designs (higher-order AR / spectrum-aware whitening); a self-test / shuffled-map control for the test-retest metrics in `17`; more subjects for the test-retest analysis; an independent language
reference map for overlap; a second software package (SPM/FSL) on one subject.

## Repo layout

```
src/fmriglm/            hrf, design, glm (OLS, AR(1), bias-corrected AR(1)), stats, mc (FDR, sign-flip max-T, clusters), data, viz, pipeline
scripts/                01 inspect, 02 validate, 03 ablation, 04 random-block null, 05 group, 06 split-half fROI, 07 quarter-shift null,
                        08 QC, 09 motion sensitivity, 10 atlas validation, 11 known-truth simulation, 12 reference check, 13 resting-state null, 14 inspect raw BIDS, 15 GLM on BIDS runs (fMRIPrep or raw), 16 QC and surrogate null on BIDS runs, 17 test-retest reliability
tests/                  synthetic unit tests
legacy_matlab_port/     Python ports of two course-lab GUIs (teaching demo only)
```

## Attribution and related work

- Language-localizer paradigm: Fedorenko et al. (2010). Teaching material that inspired the project: Idan Blank's MITx *Quantitative Biology Workshop*
  module and Kanwisher Lab data, via a course lab (handout and data are not redistributed here). Demo dataset and reference GLM: [nilearn](https://nilearn.github.io).
- Autocorrelation modelling in fMRI: Worsley et al. (2002), *NeuroImage* 15:1–15 (residual-based AR estimate with bias correction; the correction here follows the same idea);
  Woolrich et al. (2001), *NeuroImage* (flexible AR with a Tukey taper in FSL); Eklund et al. (2012), *NeuroImage* (a global AR(1) inadequate for short TRs);
  Penny et al. (2003), *NeuroImage* (variational Bayes AR order selection). Using resting-state data with fake designs as a null is an established approach in
  the false-positive literature (e.g. Eklund et al.). Verify details before citing in a formal document.

- Raw dataset: OpenNeuro ds000114 (CC0), described in the paper 'A test-retest fMRI dataset for motor, language and spatial attention functions' (GigaScience; Gorgolewski et al., 2013; verify).
  Preprocessing: fMRIPrep 25.2.5 (cite via the boilerplate in the derivatives' `logs/CITATION.md`); the container requires a free FreeSurfer license.
- Test-retest reliability of task fMRI: Bennett, C. M., & Miller, M. B. (2010), *Annals of the NY Academy of Sciences*; Elliott, M. L., et al. (2020), "What Is the Test-Retest
  Reliability of Common Task-Functional MRI Measures? New Empirical Evidence and a Meta-Analysis," *Psychological Science*, 31(7), 792-806 (mean ICC = 0.397 across 90 studies;
  ROI-level ICCs 0.067-0.485 in HCP/Dunedin samples); Kragel, P. A., et al. (2021), commentary, *Psychological Science*, 32(4), 622-626. Verify details before citing formally.

## License

Not chosen yet. Pick one before publishing (MIT is common for code).

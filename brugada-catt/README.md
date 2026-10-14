# Concept-anchored threshold transfer for Brugada syndrome ECG classifiers

Research code for a cross-site study of Brugada syndrome (BrS) ECG classification. The study uses three public cohorts and adds an external database for physician-free validation.

The main contributions are:
- **Lead restriction.** Restricting the input to V1–V3 improves cross-site AUROC.
- **A differentiable layer of 28 physiological concepts.** It includes a soft J point and is validated without physicians, on synthetic phantoms and on LUDB annotations.
- **Concept bottleneck models (CBMs)** with sign-constrained weights.
- **CATT (concept-anchored threshold transfer).** A label-free way to re-set a classifier's decision threshold at a new site. The concept model pseudo-labels the unlabelled target ECGs, and the threshold of the target model is re-chosen by Youden's J against those pseudo-labels.

The draft manuscript (Methods and Results) is in `manuscript/`. The step-by-step notes (`README_vi.md` in each folder) are in Vietnamese.

## Data (not included)

| Cohort | Source | Note |
|---|---|---|
| Brugada-HUCA | PhysioNet, `brugada-huca/1.0.0` | 363 records, 100 Hz; 356 after removing duplicates and flat-line records |
| Melo Claris and Mortara | Dryad, doi:10.5061/dryad.s1rn8pkd9 (Melo et al., *PNAS Nexus* 2023) | Patient labels reconstructed from the released scores and ROC curves (`melo_processed/reconstruct_labels.py`) |
| LUDB | PhysioNet, `ludb/1.0.1` | Used only to validate the J-point estimate |

The following files are deliberately not committed:
- harmonized arrays (`*.npz`);
- reconstructed patient labels;
- per-patient model scores (`*.jsonl`, `scores/`).

Every script regenerates its outputs from the raw downloads.

## Layout (in pipeline order)

| Folder | Content |
|---|---|
| `harmonized/` | `harmonize.py`: common format (9 leads, 100 Hz, 75-sample median beat, z-scored). `baselines.py`: ST_V2, logistic regression and CNN baselines; site classifier |
| `melo_processed/` | Label reconstruction for the Melo cohorts |
| `concepts/`, `concepts_v2/`, `concepts_v3/` | Concept layer, versions 1–3: phantoms, CBMs, prevalence stress tests |
| `concepts_v4/` | Final concept layer (`concepts3.py`), shared library (`cbm_lib4.py`), lead restriction across architectures (`arch_v4.py`), LUDB J-point validation, phantom tests |
| `concepts_v5/` | QRS-duration concept variants (`concepts5.py`), threshold transfer (`recal_v5.py`: SRC / MED / LSM / CATT / ORACLE) |
| `concepts_v6/` | Label-free comparators (EM-prior, BBSE, AdaBN, TENT) and pseudo-labeller ablation |
| `sensitivity_v7/` | Label-noise sensitivity for the Melo cohorts; device-style HUCA median beats |
| `analysis_v8/` | Reviewer-style checks: Claris/Mortara patient overlap; UPA, GMM, Otsu, DANN and Deep CORAL comparators; patient-level bootstrap; target sample size; CBM interpretation and concept decomposition of the shift; HUCA label subgroups; simulation of the CATT mechanism |
| `figures_en/` | Manuscript figures (`make_figures.py`, `make_fig6.py`), PNG at 300 dpi and PDF |
| `manuscript/` | Word draft, plus the Markdown source and the formatting script used to build it |

## Reproducing

Requirements: Python 3.10+, `numpy`, `scipy`, `pandas`, `scikit-learn`, `torch`, `matplotlib`, `wfdb`. Everything runs on CPU.

```bash
# 1. reconstruct Melo labels, then harmonize the three cohorts (see melo_processed/README_vi.md)
python harmonized/harmonize.py <huca_dir_with_metadata.csv> melo_processed harmonized
# 2. main comparisons
python concepts_v4/arch_v4.py harmonized concepts_v4
python concepts_v5/compare_v5.py harmonized concepts_v5 0 1
python concepts_v5/recal_v5.py concepts_v5 harmonized
python concepts_v6/compare_v6.py harmonized concepts_v5 concepts_v6 0 1
# 3. sensitivity analyses and step-8 checks: see sensitivity_v7/README_vi.md and analysis_v8/README_vi.md
```

Each folder's `README_vi.md` lists the exact commands and the expected run time.

## Status

The work is unpublished; please do not cite yet. Introduction, Discussion and References are in preparation.

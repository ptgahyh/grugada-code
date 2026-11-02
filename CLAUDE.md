# Grugada project context

(Written in English to save tokens: this file loads every session. Vietnamese costs noticeably more tokens for the same content.)

## User
- **Always reply, comment code and write docs in Vietnamese.** Keep English medical/statistical terms in parentheses where useful, e.g. "điểm J (J-point)".
- Goal: a scientific report/paper on ECG features for **Brugada diagnosis** and **sudden cardiac death (SCD)** using the public Brugada-HUCA dataset. Methodological correctness beats "nice" results.

## Data (verified)
- 363 ECGs, 12 leads, **100 Hz, 1200 samples (12 s)**, mV. Order I, II, III, aVR, aVL, aVF, V1–V6 (V1 = column 6).
- `metadata.csv`: `patient_id`, `basal_pattern` (0/1), `sudden_death` (0/1, **only 11 positives**), `brugada` (0/1/2; value 2 = 7 cases, meaning unknown → treat `brugada >= 1` as Brugada, 76 total, matches PhysioNet).
- Cloud sessions: physionet.org and zenodo.org are blocked. Download with `python "0.tai du lieu.py" <dir>` (raw.githubusercontent.com works), then set `BRUGADA_DIR=<dir>`. Never commit raw data.

## Code
- `ecg_pipeline.py` holds **all algorithms** (filter, QRS detection, median beat, onset/J-point, features). Numbered scripts `0`–`5` only call it. Filenames contain spaces: quote them.
- Committed outputs: `Final_Research_Data.csv`, `results/stats_*.csv`, `figures/`.
- Tests: `python -m pytest tests/` after any change to `ecg_pipeline.py`.

## Method rules (past bugs, do not repeat)
1. No hard-coded sample indices; time = seconds × `fs`.
2. Never detect R peaks on V1 alone; use `ep.detect_qrs` (12-lead energy).
3. J-point from `ep.delineate` (one global J for all leads), never "R + constant".
4. No bare `except:`; record failures in the `error` column.
5. After changing an algorithm, run step 3 on the SCD cases and **look at the plots** before trusting numbers.
6. Report effect size (AUC, median [IQR]) + 95% CI + BH q-value, not p alone. With 11 SCD cases: **no** multivariable/ML model for SCD; label results "exploratory".
7. Prefer testing literature-defined hypotheses (lead I S-wave ≥ 40 ms, aVR R ≥ 0.3 mV, wide QRS) over mass feature screening.

## Known limits
100 Hz → 10 ms resolution, no fragmented QRS; no high intercostal V1–V2; automatic J-point not yet validated against manual cardiologist measurement.

## Docs and skills
- `README.md` (how to run, changes, results), `docs/tong-quan-tai-lieu.md` (literature review with sources; **never invent citations**), `docs/skill-va-token.md` (installed skills, token tips).
- Project skills (`.claude/skills/`): `ecg-brugada-pipeline`, `small-sample-stats`, `lit-review`, `research-writing-vn` hold project-specific rules and take precedence. General skills from K-Dense (`neurokit2`, `statistical-analysis`, `scientific-visualization`, `scientific-writing`, `literature-review`, `paper-lookup`, `scientific-critical-thinking`) supply general method. `skill-scout` finds and adds missing skills from a vetted catalog when a task needs one.

## Token hygiene
- Read only the part of a file you need; don't print whole CSVs or long logs.
- Prefer one focused task per session; the user can `/clear` between tasks.

"""Count ECGs whose machine report mentions Brugada in MIMIC-IV-ECG.

Download machine_measurements.csv from PhysioNet (mimic-iv-ecg 1.0, top-level folder), then:
    python count_brugada_mimic.py machine_measurements.csv

Prints the number of ECGs and patients with a Brugada statement, the exact statements,
and writes brugada_mimic_ecgs.csv (subject_id, study_id, ecg_time, statement) for review.
Machine statements are not a diagnosis: every hit still needs review by a physician.
"""
import sys

import pandas as pd

mm = pd.read_csv(sys.argv[1], low_memory=False)
rep = [c for c in mm.columns if c.startswith("report_")]
text = mm[rep].fillna("").astype(str).agg(" | ".join, axis=1)
hit = text.str.contains("brugada", case=False)
print(f"ECGs total: {len(mm):,}   patients total: {mm.subject_id.nunique():,}")
print(f"ECGs with 'Brugada' in the machine report: {hit.sum():,}   patients: {mm.loc[hit, 'subject_id'].nunique():,}")
stmts = mm.loc[hit, rep].stack().astype(str)
print("\nStatements containing 'Brugada':")
print(stmts[stmts.str.contains("brugada", case=False)].value_counts().to_string())
out = mm.loc[hit, ["subject_id", "study_id", "ecg_time"]].copy()
out["statement"] = text[hit]
out.to_csv("brugada_mimic_ecgs.csv", index=False)
print("\nwritten brugada_mimic_ecgs.csv")

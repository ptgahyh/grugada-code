---
title: "Planning label-free threshold transfer for ECG classifiers across sites: how much the threshold moves, how many unlabelled ECGs are needed, and when not to move it"
subtitle: "Manuscript draft v5 — new and rewritten sections (to be merged into v4 Methods and Results)"
---

*Draft prepared 20 November 2026. Numbers come from the analysis folders named in brackets (e.g. [v30]); every number should be checked against those files before submission. Sections marked **NEW** do not exist in v4; sections marked **REPLACE** supersede the v4 text. Brugada numbers use the common-filter pipeline (step 27), which this draft adopts as the main pipeline; v4 numbers from the original pipeline move to a sensitivity analysis.*

# Abstract (NEW)

**Background.** ECG classifiers often keep their ranking ability when moved to a new hospital or recorder, while the decision threshold chosen at the development site stops working. Re-tuning the threshold normally needs labelled ECGs from the new site.

**Methods.** We study concept-anchored threshold transfer (CATT): an independent model labels the unlabelled target ECGs, and the threshold of the deployed model is re-chosen to maximise Youden's J against these pseudo-labels. We derive the sampling distribution of the transferred threshold, which depends only on quantities estimable without target labels, and use it to plan sample size and to decide whether the threshold should be moved at all (a "gate"). We also study a kernel-smoothed version of the threshold. We evaluate on Brugada syndrome (three cohorts, two countries, two recorders in one centre), right bundle branch block and left ventricular hypertrophy (PTB-XL and a Chinese 45,152-patient database), with analysis plans committed before each analysis and five retrainings of every model.

**Results.** Across sites, discrimination was preserved (Brugada AUROC 0.87) but the source threshold failed in 5 of 9 Brugada settings and in 2 of 6 LVH settings when the model saw millivolt input. Where it failed, CATT recovered most of the loss (balanced accuracy +0.093 [95% interval 0.080–0.106] on Brugada; +0.134 [0.120–0.148] on LVH), performing on par with unsupervised two-cluster thresholds at natural prevalence and better at low prevalence. Where the source threshold was already correct, CATT reduced balanced accuracy (−0.013 to −0.038). The predicted spread of the threshold matched observation (ratio 1.07–1.09 on Brugada and LVH; 0.81 on RBBB). Smoothing halved the spread and cut the number of unlabelled ECGs needed 2–8-fold. The gate preserved all benefit on Brugada but did not prevent harm on large target cohorts. Patient-level bootstrap intervals understated uncertainty 1.2–1.7-fold relative to retraining variability.

**Conclusions.** Label-free threshold transfer can be planned in advance, and its precision predicted, but whether it helps depends on whether the source threshold is broken, which current label-free tests detect only for small target cohorts.

**Keywords:** electrocardiogram; domain shift; decision threshold; Brugada syndrome; unsupervised adaptation; cube-root asymptotics

# 1. Introduction (NEW)

Deep learning models for the electrocardiogram (ECG) are increasingly trained at one hospital and deployed at others. A recurring finding is that discrimination, measured by the area under the ROC curve, transfers better than calibration and operating points: a model that ranks patients correctly at a new site may still classify them poorly because its output scale has shifted [refs on ECG domain shift, to add]. For a binary decision this shift shows up as a wrong threshold. Re-choosing the threshold is cheap if the new site can label a few hundred ECGs, but labelling requires expert reading and, for rare conditions, many ECGs before enough positives are seen.

Brugada syndrome (BrS) illustrates the problem sharply. Its diagnostic ECG pattern (coved ST elevation in the right precordial leads) is often absent at baseline and appears only after sodium-channel-blocker challenge [Antzelevitch 2016]. Recent deep learning work has shown that baseline ECGs carry a learnable signature [Melo 2023], but public BrS cohorts are small, come from few centres, and differ in recorder, lead placement and preprocessing. A model trained on one cohort therefore faces exactly the deployment situation above.

Label-free alternatives exist. Unsupervised thresholding treats the score distribution as a two-component mixture (two-component Gaussian mixtures, Otsu's method [Otsu 1979]); label-shift methods re-estimate class priors [Saerens 2002; Lipton 2018]; test-time adaptation updates the network itself [Li 2018; Wang 2021]. These methods return a threshold or an adapted model, but none tells the user how precise that threshold is, how many unlabelled ECGs are needed, or whether moving the threshold is worth the risk.

This paper studies a simple transfer rule, concept-anchored threshold transfer (CATT): an independent "anchor" model, here a concept bottleneck model built on physiological measurements, pseudo-labels the target ECGs, and the deployed model's threshold is re-chosen to maximise Youden's J against the pseudo-labels. Our focus is not to propose CATT as a better adaptation method, but to make it plannable. We make four contributions.

1. **Distribution of the transferred threshold.** We show that the CATT threshold has a cube-root (Chernoff-type) limit whose scale depends only on two quantities of the pseudo-labelled score distribution. Both are estimable without target labels, so a site can predict the precision of its threshold, and the number of unlabelled ECGs needed, before any labelling. The result follows from standard argmax theory [Kim and Pollard 1990]; our contribution is to identify the label-free constants and turn them into a sample-size rule.
2. **A smoothed threshold** whose spread is about half as large and which needs 2–8 times fewer ECGs for the same precision, with a closed-form spread that reuses the same two constants.
3. **A gate**: move the threshold only when it differs from the source threshold by more than its predicted sampling noise.
4. **An honest evaluation**: three diagnostic tasks and four public cohorts; every analysis after the first round was specified in a plan committed to a public repository before it was run; all pre-specified criteria are reported, including those not met; and every model was retrained five times to measure variability that patient-level bootstrap intervals ignore.

The results support the planning tools more than the transfer rule itself. CATT recovers most of the loss when the source threshold is broken, but it is not better than unsupervised two-cluster thresholds at natural prevalence, and it harms performance when the source threshold was already correct. The gate prevents this harm only when the target cohort is small. We argue that these limits, rather than an average gain, are what a site needs to know before relying on label-free threshold transfer.

# 2. Methods: additions and corrections

## 2.2 Signal harmonization (REPLACE the filtering sentence)

> v4 states that "all ECGs were band-pass filtered". This was true only for HUCA. Replace with:

Brugada-HUCA recordings were band-pass filtered (0.5–40 Hz, second-order Butterworth, zero phase) before beat extraction. The Melo representative beats, which are device-computed single beats, were low-pass filtered with the same 40 Hz filter at their native 200 Hz before resampling; a 0.5 Hz high-pass cannot be applied meaningfully to a single 750-ms beat. Results with the original pipeline, in which only HUCA was filtered, are reported as a sensitivity analysis (Section 3.11).

## 2.x Right bundle branch block and left ventricular hypertrophy (REPLACE data description where needed)

We used PTB-XL v1.0.3 [Wagner 2020], split by recorder (CS and AT devices), and the PhysioNet "large scale 12-lead ECG database for arrhythmia study" v1.0.0 from Shaoxing and Ningbo [Zheng 2022; Zheng 2020] ("cn"). RBBB was defined in cn by either of two SNOMED codes (713427006, 59118001); LVH by 164873001. Our copy of the cn database was obtained through a third-party mirror that duplicated records and truncated some signal files; reading every copy of each record recovered 45,127 of 45,152 records, and the analysed cohorts (RBBB 43,380 records, 1,733 positive; LVH 38,221, 637 positive) differ from those obtained from the official release on 11 November (43,403 and 38,242) by 23 and 21 records. These two tasks, the six transfer settings for each (CS↔AT, CS↔cn, AT↔cn), the anchors and all success criteria were registered before their first run.

For LVH we report two target models: the standard model, which receives lead-wise z-scored beats like all other models, and a model receiving beats in millivolts (Section 3.14). The z-scored model cannot see amplitude, which is the main LVH criterion.

## 2.y Smoothed CATT threshold (NEW)

Instead of maximising the empirical Youden curve, which is a step function, the smoothed threshold t̂_h is the point where the two Gaussian-kernel density estimates of the deployed score within the pseudo-positive and pseudo-negative groups cross. The bandwidth is h = c·(n·m)^(−1/5), with c = 0.9·min(SD, IQR/1.34) of the scores and m = min(q, 1 − q), q being the pseudo-positive fraction. By a standard delta-method expansion of kernel estimators [Horowitz 1992],

  SD(t̂_h) ≈ √(R·Ṽ/(n·h)) / Ã,  R = 1/(2√π),

with the same constants Ṽ and Ã as in Theorem 1. Requiring SD ≤ ε gives the sample size n ≥ [R·Ṽ·m^(1/5)/(c·Ã²·ε²)]^(5/4), compared with n ≥ 0.54·Ṽ/(Ã²ε³) for the unsmoothed threshold. In practice we estimate SD(t̂_h) from one unlabelled sample with a sandwich (influence-function) estimator ("G2"). The sample-size rule is applied with a fixed safety factor of 1.5, chosen from simulations before the RBBB/LVH analyses.

## 2.z Gate (NEW)

The gated threshold keeps the source threshold t_src unless |t_CATT − t_src| > z·σ̂, where σ̂ is the label-free estimate of SD(t_CATT) (Theorem 1 for the unsmoothed threshold, G2 for the smoothed one). We pre-specified z = 2, with z = 1 and z = 3 as sensitivity analyses.

## 2.w Uncertainty from retraining (NEW)

Patient-level bootstrap intervals treat the trained models as fixed. To measure the additional variability from training, every trained component (target CNNs, the independent CNN anchor, concept bottleneck anchors, DANN and CORAL) was retrained five times with different random seeds, keeping data, validation splits and analyses fixed. For each comparison we report the mean over the five retrainings and a combined 95% interval, mean ± 1.96·√(SE_boot² + s²/5), where SE_boot is the median bootstrap standard error of a single retraining and s the standard deviation across retrainings. The pre-registered rule for this analysis (a t-interval over retrainings plus sign consistency) ignores patient-level uncertainty, because all retrainings use the same patients; we noticed this after seeing the results and report both rules (Section 3.15).

## 2.v Pre-registration and reporting (NEW)

From the RBBB/LVH analyses onwards (steps 14–30 in the repository), each analysis had a written plan with explicit success criteria, committed to the public repository before it was run. Deviations and post-hoc observations are labelled as such in the text. Patient-level scores and labels are not published; aggregated results and all code are.

# 3. Results: new sections

## 3.10 Why the threshold fails: where the model looks and what changes between sites (NEW)

*Note: the attribution analysis below uses the original pipeline; its code must be re-run on the common-filter pipeline before submission (repository step 17).*

The models attend to the same region as clinicians. Occlusion maps concentrated on the J point and early ST segment of V1–V2, which received 3.8 times more importance than its share of the beat (range 3.0–4.5 over 9 settings; integrated gradients 2.8–4.5). Maps computed on source and target data were nearly identical (median cosine 0.95, range 0.91–0.99), and models trained in Spain and in Italy agreed less (0.80–0.87) than models trained on two recorders in the same Italian centre (0.96). Threshold failure is therefore not caused by the model looking at the wrong place in the new site's ECGs. On HUCA, where millivolt amplitudes are available, 44% of controls had a J-point elevation of at least 0.2 mV in V1 or V2, so an amplitude criterion alone would not be specific; the models, which see z-scored input, rely on shape.

The shift is a change of scale across the whole target cohort, not a change in case mix. Clustering V1–V2 waveforms without labels (K = 5) showed that 97% (95% CI 93–103%) of the mean logit shift of the negative class occurred within the same waveform cluster; re-weighting cluster composition alone closed 4% of the gap to the oracle threshold, compared with 91% for CATT. Waveform composition did differ between centres (total variation 0.67 between HUCA and Claris, 0.24 between the two Italian recorders).

Two acquisition explanations were tested and accounted for only a minority of the shift. A 20-ms difference in the position of the S-wave nadir relative to the alignment point (a post-hoc observation) closed 1% of the gap when corrected at scoring time, and retraining with a common alignment left all five failing settings failing. Matching the spectral content of the Italian beats to HUCA closed 21% of the gap at scoring time and raised balanced accuracy at the source threshold from 0.567 to 0.620 after retraining, again with all five settings still failing. An earlier simulation of the acquisition chain explained about one third of the shift. Preprocessing harmonisation therefore helps, but most of the shift remains and has to be handled at the threshold.

## 3.11 Main Brugada results under a common filtering pipeline (NEW)

With the same 40-Hz low-pass filter applied to every cohort, the main results were unchanged. AUROC across sites was 0.868; the source threshold failed in the same 5 of 9 settings; balanced accuracy in the held-out settings was 0.703 at the source threshold, 0.800 with CATT and 0.823 at the oracle threshold, a CATT gain of +0.097 (95% CI 0.085–0.111) [v27]. Rerunning every subsequent Brugada analysis on the new scores (13 analyses, 580 compared quantities) changed the status of no pre-specified criterion [v29]. The filter narrowed only part of the spectral difference (energy above 15 Hz in V1–V3: Claris 0.204 → 0.186, HUCA 0.147).

One comparison did change: against unsupervised two-cluster thresholds at natural prevalence, CATT was no longer significantly better (vs two-component Gaussian mixture +0.006, −0.004 to 0.015; vs Otsu +0.004, −0.004 to 0.015). At 5% prevalence, obtained by subsampling positives, CATT remained better (+0.018 and +0.021). Two-cluster methods assume a bimodal score distribution, which disappears when positives are rare.

## 3.12 The smoothed threshold (NEW)

Smoothing roughly halved the spread of the transferred threshold on Brugada (median ratio of SDs 0.56 [v29]; 0.54 in the original pipeline) and reduced it by about one third in simulation (RMSE ratio 0.65), with the expected faster rate (slope of log SD on log n −0.41 vs −0.31). The closed-form spread matched simulation (median predicted/observed 0.98; 0.93–1.00 across n) and held under skewed, heavy-tailed, bimodal and discretised score distributions (0.95–1.00) [v24, v25]. Accuracy did not suffer: with 50 unlabelled ECGs, mean balanced accuracy was 0.796 vs 0.789 and the 10th percentile 0.774 vs 0.751; on the full cohort the two thresholds were equal (0.801 vs 0.800) [v29]. For the same precision the smoothed threshold needed far fewer ECGs (Brugada, ε = 0.30 logit: 69 vs 411; ε = 0.15: 389 vs 3290).

On RBBB and LVH the reduction in spread replicated (SD ratio 0.78, 0.53 and 0.46 for RBBB, LVH and LVH with millivolt input), but the label-free spread estimate and the sample-size rule did not replicate on RBBB [v28]. The single-sample estimate G2 was accurate on LVH (median predicted/observed 0.94; 0.97 with millivolt input) and on Brugada (0.90), but underestimated the spread five-fold on RBBB (0.22). Correspondingly, the sample-size rule with the pre-specified safety factor of 1.5 reached its target in 87–98% of plans on LVH but only 47–58% on RBBB (criterion: ≥ 80%). Post hoc, RBBB is the rarest target (2.4–4.0% prevalence) and the best separated (balanced accuracy 0.97); with 500 ECGs there are only 10–20 pseudo-positives, and the ratio of predicted to observed spread rose with n (e.g. 0.31, 0.61, 0.81 at n = 500, 1000, 2000 in one setting), consistent with an asymptotic approximation applied too early. We therefore present the sample-size rule as valid only when the expected number of pseudo-positives is not small, a condition not tested in advance.

## 3.13 When not to move the threshold: the gate (NEW)

On Brugada the gate behaved as intended. In the five failing settings the CATT threshold lay 6.6–24 estimated SDs from the source threshold, so the gate always opened and kept the full benefit; in the non-failing settings the distance was 1.2–1.7 SDs and the gate stayed closed in 8–9 of 12 runs [v26, v29]. In terms of labelled data, CATT was worth about 50 labelled target ECGs at natural prevalence and about 100 at 5% prevalence.

On LVH the gate failed its pre-specified criterion (gated minus source balanced accuracy −0.008, −0.012 to −0.005; criterion ≥ −0.003) [v28]. With thousands of target ECGs, the CATT threshold differed from the source threshold by 3–37 SDs in every setting, so the gate was nearly always open (15 of 18 runs), including where CATT moved the threshold in the wrong direction. The gate tests sampling noise; it cannot detect bias of the anchor, which does not shrink with n. It is useful when the target cohort is small, and insufficient when it is large.

## 3.14 A second task with a failing threshold: LVH with millivolt input (NEW)

When the LVH target model received beats in millivolts rather than z-scored beats, its AUROC improved in all six settings (by 0.04–0.08), and the source threshold failed in two of them, both from the Chinese cohort to PTB-XL (balanced accuracy 0.728 vs oracle 0.872, and 0.699 vs 0.845) [v22]. In these two settings CATT recovered most of the loss (+0.134 over five retrainings, 0.120–0.148), on par with Gaussian-mixture and Otsu thresholds (−0.004 and −0.007, intervals including 0). In the four settings where the source threshold was correct, CATT reduced balanced accuracy by 0.038 (0.034–0.042). Theorem 1 predicted the spread of the threshold (ratio 1.07, Spearman 0.97; label-free 1.15).

With z-scored input, no LVH or RBBB setting failed. CATT then changed balanced accuracy by −0.013 (LVH, −0.018 to −0.007) and +0.011 (RBBB, 0.007 to 0.015) relative to the source threshold, and was better than two-cluster thresholds on LVH (+0.031, +0.037), where positives are rare (1.7–10%).

## 3.15 Variability from retraining (NEW)

Retraining every model five times left the qualitative results unchanged [v30]. The number of failing settings was identical in all five retrainings for Brugada (5) and LVH with millivolt input (2), and zero for LVH; RBBB had one failing setting in one retraining. Table X gives each comparison.

Retraining variability was as large as patient-sampling variability or larger. The between-retraining SD of the CATT gain divided by the bootstrap standard error of a single retraining was 1.25 (RBBB), 1.49 (LVH) and 1.35–1.68 (LVH with millivolt input), and 1.8–2.8 for comparisons with two-cluster thresholds; only on Brugada was it at or below 1 (0.99). Intervals in this paper therefore combine both sources.

Under the pre-registered rule, CATT was judged better than Gaussian-mixture and Otsu thresholds on Brugada at natural prevalence (five retrainings, all positive, mean +0.009 and +0.008). Under the combined interval it was not (−0.002 to 0.020 and −0.002 to 0.019); the two rules agreed on every other comparison. We report the combined interval as primary because it accounts for patient sampling.

One discrepancy remains unexplained. The first RBBB run (11 November, official PhysioNet release, different machine and software versions) gave CATT minus source −0.008 (−0.013 to −0.005), outside the range of all five later retrainings (+0.007 to +0.013). The difference therefore reflects the computing environment or data copy rather than the random seed. We do not interpret the RBBB effect.

**Table X. CATT versus comparators: mean over five retrainings and combined 95% interval** [v30/results/summary_table.csv]

| Task (settings) | vs source threshold | vs 2-component GMM | vs Otsu |
|---|---|---|---|
| Brugada, held-out, natural prevalence | +0.093 (0.080, 0.106) | +0.009 (−0.002, 0.020) | +0.008 (−0.002, 0.019) |
| Brugada, held-out, 5% prevalence | +0.092 (0.073, 0.112) | +0.020 (0.006, 0.034) | +0.024 (0.009, 0.038) |
| LVH, mV input, 2 failing settings | +0.134 (0.120, 0.148) | −0.004 (−0.015, 0.007) | −0.007 (−0.019, 0.005) |
| LVH, mV input, 4 non-failing settings | −0.038 (−0.042, −0.034) | +0.023 (0.017, 0.029) | +0.025 (0.019, 0.031) |
| LVH, z-scored input, all 6 | −0.013 (−0.018, −0.007) | +0.031 (0.026, 0.036) | +0.037 (0.032, 0.043) |
| RBBB, all 6 | +0.011 (0.007, 0.015) | +0.007 (0.004, 0.011) | −0.008 (−0.011, −0.005) |

# 4. Discussion (NEW)

**Summary.** Across three ECG tasks, moving a model to a new site or recorder preserved discrimination, but the decision threshold failed in some transfers: often for Brugada syndrome between Spain and Italy, and for LVH between China and Germany once the model could see voltage. In those transfers a threshold chosen against pseudo-labels from an independent model recovered most of the lost balanced accuracy without any target labels. The precision of this threshold was predictable from unlabelled data, and smoothing it halved its spread. The same procedure lowered accuracy wherever the source threshold had been correct, and a statistical gate prevented this only for small target cohorts.

**What the theory adds.** Unsupervised threshold selection is not new, and on the main Brugada comparison CATT was not better than a two-component mixture or Otsu's method at natural prevalence. What these methods lack is a statement of precision. The cube-root limit of the transferred threshold, with constants that depend only on the pseudo-labelled score distribution, lets a site state before labelling how variable its threshold will be and how many unlabelled ECGs it needs. This prediction held on Brugada, LVH and a non-ECG tabular task (ratio of predicted to observed spread 1.06–1.12) and was less accurate on RBBB (0.80); its smoothed counterpart held on Brugada and LVH. It failed on RBBB, where positives were rarest and separation strongest; the most plausible reason is that the asymptotic regime requires more pseudo-positives than a few hundred ECGs provide. A minimum number of expected pseudo-positives should therefore accompany the sample-size rule; we did not test such a condition in advance and leave it to future work.

**Variance is not bias.** The central limitation is that the theory describes the sampling noise of the transferred threshold around its own target, not the distance between that target and the correct threshold. That distance is set by how the anchor's errors relate to the deployed model's errors. When the source threshold fails badly, any reasonable anchor moves the threshold in the right direction and the gain is large. When the source threshold is already correct, an imperfect anchor can only move it away. The gate exploits the first fact: it moves the threshold only if the move is large relative to sampling noise. On small target cohorts, such as Brugada, a correct threshold produces moves of one or two standard errors and the gate stays closed. On cohorts of thousands, any anchor bias becomes "significant" and the gate opens. A useful label-free gate for large cohorts would need to compare the move to an estimate of anchor bias, for example by checking agreement between two independent anchors; this remains open.

**Where the shift comes from.** Several analyses converge on the same picture for Brugada. The models look where clinicians look, and they look at the same place at both sites. The shift is a change in the scale of the scores across the whole cohort rather than a different mix of waveform types. Alignment differences do not explain it; spectral differences explain about a fifth, and a simulated acquisition chain about a third. The representative beats of the Italian cohorts are computed by the recorders themselves, while the HUCA beats are medians of filtered raw beats; such device-level processing is hard to undo and is a plausible source of the remainder. This argues for treating the threshold as something to re-establish at each site rather than something preprocessing can make invariant.

**Retraining variability.** Bootstrap intervals that hold the trained model fixed were too narrow by a factor of 1.2–1.7 on RBBB and LVH, and nearly two for small differences between methods. Single-run comparisons of adaptation methods with differences of about 0.01 should therefore be read with caution. Our own first RBBB result changed sign between computing environments, by an amount larger than the seed-to-seed variation. Reporting results over several retrainings, and stating which sources of variability an interval covers, is a simple safeguard.

**Clinical relevance.** We chose thresholds by balanced accuracy, a symmetric criterion. Screening for Brugada syndrome would favour sensitivity, and fixed-sensitivity thresholds are quantiles with a different (square-root) rate that our theory does not cover. The Brugada cohorts are enriched (20–52% positive); at 5% prevalence CATT kept its gain over the source threshold and became better than two-cluster methods, which is encouraging for low-prevalence settings but was obtained by subsampling, not in a screening population. In terms of labelled data, CATT was worth about 50 labelled ECGs at natural prevalence and 100 at 5%; a site that can label more should do so.

**Limitations.** (i) Evidence that CATT helps comes from two transfers per task at most (Spain↔Italy for Brugada, China→Germany for LVH with millivolt input). (ii) The Italian Brugada labels were reconstructed from published ROC curves, and the label definitions differ between cohorts: most Italian positives were confirmed by ajmaline, and 71% of HUCA positives had a non-diagnostic baseline ECG. (iii) No clinician reviewed the ECGs; concept validation used synthetic phantoms and annotated J points. (iv) The z-scored input hides amplitude, so the standard LVH model could not use the main LVH criterion. (v) The cn data came from a third-party mirror; 23 (RBBB) and 21 (LVH) records were lost relative to the official release, and the first RBBB run on the official release differed in sign. (vi) The attribution analysis (Section 3.10) has not yet been rerun on the common-filter pipeline. (vii) The safety factor of the sample-size rule was chosen from simulations, and the rule failed on RBBB. (viii) The theory covers Youden thresholds only.

# 5. Conclusion (NEW)

Label-free threshold transfer for ECG classifiers can be planned: the precision of the transferred threshold and the number of unlabelled ECGs needed can be predicted from unlabelled data, and smoothing the threshold roughly halves its spread. Whether the transfer helps depends on whether the source threshold was broken. When it was, CATT recovered most of the loss; when it was not, CATT did harm, and a statistical gate prevented this only for small target cohorts. Detecting anchor bias without labels is the main open problem for safe deployment.

# Data and code availability (NEW)

All public datasets are cited. Code, analysis plans with commit times, and aggregated results are available at github.com/ptgahyh/grugada-code (branch to be fixed at submission). Patient-level scores are not shared.

# References (draft; verify every entry before submission)

1. Antzelevitch C, Yan GX, Ackerman MJ, et al. J-Wave syndromes expert consensus conference report: emerging concepts and gaps in knowledge. Heart Rhythm. 2016;13(10):e295–e324. *(verify)*
2. Melo L, Ciconte G, Christy A, Vicedomini G, Anastasia L, Pappone C, Grant E. Deep learning unmasks the ECG signature of Brugada syndrome. PNAS Nexus. 2023;2(11):pgad327. doi:10.1093/pnasnexus/pgad327. Data: Dryad doi:10.5061/dryad.s1rn8pkd9.
3. Costa Cortez N, Garcia Iglesias D. Brugada-HUCA: 12-lead ECG recordings for the study of Brugada syndrome (v1.0.0). PhysioNet. 2026. doi:10.13026/0m2w-dy83. *(verify authors/title)*
4. Wagner P, Strodthoff N, Bousseljot RD, et al. PTB-XL, a large publicly available electrocardiography dataset. Sci Data. 2020;7:154. *(verify)*
5. Zheng J, Guo H, Chu H. A large scale 12-lead electrocardiogram database for arrhythmia study (v1.0.0). PhysioNet. 2022. doi:10.13026/wgex-er52.
6. Zheng J, Zhang J, Danioko S, Yao H, Guo H, Rakovski C. A 12-lead electrocardiogram database for arrhythmia research covering more than 10,000 patients. Sci Data. 2020;7:48. doi:10.1038/s41597-020-0386-x.
7. Kalyakulina AI, Yusipov II, Moskalenko VA, et al. LUDB: a new open-access validation tool for electrocardiogram delineation algorithms. IEEE Access. 2020;8:186181–186190. *(verify)*
8. Kim J, Pollard D. Cube root asymptotics. Ann Stat. 1990;18(1):191–219. *(verify pages)*
9. Groeneboom P, Wellner JA. Computing Chernoff's distribution. J Comput Graph Stat. 2001;10(2):388–400. *(verify)*
10. Banerjee M, McKeague IW. Confidence sets for split points in decision trees. Ann Stat. 2007;35(2):543–574. *(verify)*
11. Horowitz JL. A smoothed maximum score estimator for the binary response model. Econometrica. 1992;60(3):505–531. *(verify)*
12. Hong H, Li J. The numerical bootstrap. Ann Stat. 2020;48(1):397–412. doi:10.1214/19-AOS1812.
13. Cattaneo MD, Jansson M, Nagasawa K. Bootstrap-based inference for cube root asymptotics. Econometrica. 2020;88(5):2203–2219. doi:10.3982/ECTA17950.
14. Youden WJ. Index for rating diagnostic tests. Cancer. 1950;3(1):32–35. *(verify)*
15. Otsu N. A threshold selection method from gray-level histograms. IEEE Trans Syst Man Cybern. 1979;9(1):62–66. *(verify)*
16. Saerens M, Latinne P, Decaestecker C. Adjusting the outputs of a classifier to new a priori probabilities: a simple procedure. Neural Comput. 2002;14(1):21–41. *(verify)*
17. Lipton ZC, Wang YX, Smola AJ. Detecting and correcting for label shift with black box predictors. ICML 2018. *(verify)*
18. Ganin Y, Ustinova E, Ajakan H, et al. Domain-adversarial training of neural networks. J Mach Learn Res. 2016;17(59):1–35. *(verify)*
19. Sun B, Saenko K. Deep CORAL: correlation alignment for deep domain adaptation. ECCV Workshops 2016. *(verify)*
20. Li Y, Wang N, Shi J, Hou X, Liu J. Adaptive batch normalization for practical domain adaptation. Pattern Recognit. 2018;80:109–117. *(verify)*
21. Wang D, Shelhamer E, Liu S, Olshausen B, Darrell T. Tent: fully test-time adaptation by entropy minimization. ICLR 2021. *(verify)*
22. Menon AK, van Rooyen B, Ong CS, Williamson RC. Learning from corrupted binary labels via class-probability estimation. ICML 2015. *(verify; cited in v4 for the invariance result)*
23. [To add: 2–3 references on ECG model performance under site/recorder shift.]

# Notes for the next writing pass (not part of the manuscript)

- Merge order into v4: Abstract and Introduction at the top; 2.2 sentence replaced; 2.x, 2.y, 2.z, 2.w, 2.v inserted after the existing 2.10; Sections 3.10–3.15 after 3.9; Discussion and Conclusion at the end.
- Replace v4 Brugada headline numbers with the common-filter values (Section 3.11) and move v4 values to a supplementary table.
- Replace the finite-population exponent 1/3 with the empirical 0.24 (step 18a) wherever it appears; the smoothed threshold uses 0.45.
- Move F1 (half-sample estimator), the bootstrap variants and the fixed-sensitivity extension to the supplement, as recommended in the 14/11 review.
- Figures needed: (a) attribution map (rerun step 17), (b) CATT gain vs whether the source threshold failed, all tasks, with retraining intervals, (c) predicted vs observed spread, ordinary and smoothed, four datasets, (d) gate distance in SD units, Brugada vs LVH.
- Check that the target journal (CIBM or IEEE JBHI) word limit fits; the current Results are long.

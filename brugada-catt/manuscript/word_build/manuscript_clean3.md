---
title: "Concept-anchored threshold transfer for label-free cross-site deployment of Brugada syndrome ECG classifiers"
subtitle: "Manuscript draft — Methods and Results"
---


*Draft of Sections 2 (Methods) and 3 (Results). Introduction, Discussion and References to follow.*

# 2. Methods

Figure 1 summarizes the pipeline.

![](fig1_overview.png){width=16cm}

**Figure 1.** Overview. (A) Three Brugada cohorts are harmonized to a common nine-lead median-beat format. Two models are trained on labelled source sites: a CNN restricted to V1–V3 (the target model f), and a sign-constrained concept bottleneck model g built on a differentiable layer of 28 physiological measurements. The measurements are validated without physicians, on synthetic phantoms and against J-point annotations in LUDB. (B) At an unlabelled target site, g pseudo-labels the batch at its own source threshold, and the threshold of f is re-chosen to maximize Youden's J against those pseudo-labels (concept-anchored threshold transfer, CATT). Neither target labels nor changes to f are needed.

## 2.1 Datasets

We used three public Brugada syndrome (BrS) ECG cohorts, treated as three domains (sites), and one public ECG database with cardiologist annotations for physician-free validation of the measurement layer (Table 1).

**Brugada-HUCA** (Hospital Universitario Central de Asturias, Spain; PhysioNet) contains 363 standard 12-lead recordings of 12 s sampled at 100 Hz. Each record carries three labels:

- a diagnostic label for Brugada syndrome (`brugada`);
- a flag for a Brugada-type pattern on the baseline ECG (`basal_pattern`);
- a sudden-death flag.

The diagnostic label takes the value 2 in seven records, a value that the database README describes only as "other/atypical cases". We treated labels 1 and 2 as BrS-positive, and checked that excluding these seven records does not change the results (Section 3.9).

We removed four exact duplicates (patient IDs 844915, 1220716, 1366050, 1307673; the first copy was kept) and three records containing flat-line segments of at least 2 s in at least one lead (267630, 903802, 1230482). None of the removed records belonged to a patient with sudden death. The final cohort has 356 recordings, 73 of them BrS-positive (20.5%).

Only 21 of the 73 positives show a Brugada-type pattern on the baseline ECG, whereas 23 negatives do. Most HUCA positives, like most Melo positives, therefore do not show a spontaneous diagnostic pattern on the analysed ECG.

**Melo-Claris and Melo-Mortara** (Melo et al., 2023; Dryad doi:10.5061/dryad.s1rn8pkd9) come from one Italian centre and were acquired on two different recorders (Abbott Workmate Claris and Mortara ELI 350). Each patient is represented by one device-computed representative beat of 150 samples at 200 Hz (750 ms), with every lead z-scored by the authors. The diagnosis was confirmed by ajmaline testing, and about 83% of positives showed a type-1 pattern only after drug challenge. Claris contains 1154 patients (596 positive, 51.6%) and Mortara 370 (137 positive, 37.0%). For Mortara we kept the nine segments that matched the Claris lead order (I, II, III, V1–V6).

**Label provenance.** The Dryad release contains no patient-level label file. We reconstructed labels from the released per-patient prediction scores (mean of the seven-model ensemble) and the published ROC curves. Patients were sorted by score; each vertical step of the ROC was assigned to positives and each horizontal step to negatives. The reconstruction reproduces the reported class counts and the reported true-positive and true-negative counts at the 0.5 threshold exactly, for the corresponding result files (Claris 509/460; Mortara, CNN model, 102/207). For Mortara, labels reconstructed from the CNN and DNN result files agreed on 361 of 361 comparable patients. We assess the impact of residual reconstruction error in Section 2.10.

**LUDB** (Lobachevsky University Electrocardiography Database v1.0.1) contains 200 10-s 12-lead recordings at 500 Hz, with P, QRS and T onsets and offsets annotated beat by beat by cardiologists. Nine recordings with pacemakers were analysed separately. LUDB contains no BrS labels and was used only to validate the J-point estimate.

**Table 1. Cohorts.**

| Cohort | Country, recorder | Records | BrS-positive | Native format | Role |
| --- | --- | --- | --- | --- | --- |
| Brugada-HUCA | Spain, 12-lead recorder (100 Hz) | 356 | 73 (20.5%) | 12-s raw 12-lead | Training and testing |
| Melo-Claris | Italy, Abbott Claris | 1154 | 596 (51.6%) | Representative beat, 200 Hz, z-scored | Training and testing |
| Melo-Mortara | Italy, Mortara ELI 350 | 370 | 137 (37.0%) | Representative beat, 200 Hz, z-scored | Training and testing |
| LUDB | Russia | 200 (9 paced) | — | 10-s raw 12-lead, 500 Hz | J-point validation only |

## 2.2 Signal harmonization

All cohorts were mapped to one format: one representative beat per patient, nine leads (I, II, III, V1–V6), 100 Hz, 75 samples with the QRS energy peak at sample 28 (−280 ms to +460 ms around it), each lead z-scored. We chose 100 Hz because it is the native rate of Brugada-HUCA. We kept nine leads because they are the leads available in both Melo cohorts.

**HUCA.** Leads were converted to mV and band-pass filtered (second-order Butterworth, 0.5–40 Hz, applied forward and backward for zero phase). QRS complexes were detected as peaks of the summed absolute first difference across the nine leads (minimum distance 330 ms, height ≥ 35% of the 99th percentile). Beats were cut from −290 ms to +450 ms around each peak (75 samples), and the sample-wise median over all complete beats (at least three) formed the representative beat.

**Melo.** The nine leads of each 200 Hz beat were resampled to 100 Hz with a polyphase filter with anti-aliasing (150 to 75 samples).

**Common steps.** Each beat was shifted so that the peak of its QRS energy, defined as the summed absolute first difference across leads, falls at sample 28, padding with edge values. Each lead was then z-scored. Z-scoring removes amplitude units, which are absent from the Melo data, and with them any amplitude-scale shortcut.

To quantify the site information that survives harmonization, we trained a linear site classifier: L2 logistic regression (C = 0.01) on the 675 raw samples, evaluated by five-fold cross-validation.

## 2.3 Evaluation protocol

**Cross-site settings.** With three domains we formed nine out-of-domain settings: the six single-source pairs (train on one cohort, test on another) and the three leave-one-domain-out settings (train on two cohorts, test on the third). The target cohort is never used for training, model selection or threshold selection. For reference, within-domain performance was estimated by five-fold stratified group cross-validation. Three near-identical beat pairs in Claris (correlation > 0.999, same label) were kept in the same fold.

**Validation splits.** In every setting the training pool was split 85/15 into training and validation sets, stratified jointly by source domain and label. The split was repeated with three seeds. The validation set was used for early stopping and for choosing the decision threshold (Youden's J). All results are aggregated over the three split seeds.

**Metrics.**

- *Discrimination:* AUROC of the target scores averaged over the three split seeds.
- *Operating point:* sensitivity, specificity and balanced accuracy (their mean), computed for each split seed and then averaged. Balanced accuracy is the quantity a threshold-transfer method must preserve, and it does not depend on prevalence.
- *Summaries:* means across settings, plus the worst setting, because a deployable method must not fail at any site.

**Statistics.**

- *AUROC comparisons* used a paired bootstrap of target patients: 2000 resamples, two-sided p = 2·min(P(Δ ≤ 0), P(Δ ≥ 0)). The nine settings of each comparison were corrected with the Holm method, and we report wins, ties and losses at Holm-adjusted p < 0.05.
- *Threshold methods* were compared by the Wilcoxon signed-rank test across settings.

**Prevalence stress test.** Screening populations have far lower BrS prevalence than referral cohorts. We therefore subsampled the positives of each target cohort to 5%, 10% and 20% prevalence (10 random draws each), keeping all negatives, and also report the natural prevalence. When a cohort had too few positives for the requested prevalence, negatives were subsampled instead.

## 2.4 Baseline classifiers and lead restriction

**ST\_V2 rule.** A single hand-crafted feature without training: the mean amplitude of V2 from 60 to 90 ms after the QRS energy peak, minus the mean over the PR interval (−140 to −90 ms); both windows include their end points (samples 34–37 and 14–19 of the aligned beat). Its threshold was chosen on the validation set like every other model.

**Logistic regression.** L2-regularized logistic regression on the standardized raw samples (C ∈ {10⁻⁴, 10⁻³, 10⁻², 10⁻¹} by five-fold cross-validated AUROC, class-balanced weights).

**CNN.** A one-dimensional CNN with three convolution–batch-normalization–ReLU blocks (32, 64 and 64 filters; kernels 7, 5 and 5; max-pooling by 2 after the first two blocks), global average pooling, dropout 0.3 and a linear output.

- *Training:* Adam (learning rate 10⁻³, weight decay 10⁻³), batch size 64, binary cross-entropy weighted by the class ratio, at most 60 epochs, early stopping after 12 epochs without improvement in validation AUROC.
- *Augmentation:* random circular shifts of ±2 samples, per-lead gain of ±10% and Gaussian noise (σ = 0.05).
- *Ensemble:* three seeds, with predicted probabilities averaged.

**ResNet1D.** Three residual blocks (32, 64 and 64 channels; kernels 7, 5 and 3), each made of two convolutions with batch normalization and a 1 × 1 projection shortcut when the width changes. It was trained exactly like the CNN.

**Lead restriction.** Each learned model was trained twice: on all nine leads, and on V1–V3 only (a 3 × 75 input). V1–V3 are the right precordial leads on which BrS diagnostic criteria are defined. Restricting the input removes the leads on which a model can learn site-specific signal unrelated to the disease.

## 2.5 Differentiable Brugada concept layer

The concept layer maps a representative beat x ∈ ℝ^9×75^ to 28 measurements that mirror how BrS ECGs are read: where the QRS complex ends, how high the ST segment takes off in V1–V3, and whether it descends in a coved shape. Every operation is differentiable, so five fiducial parameters can be learned jointly with the classifier.

Design constraints:

- *Affine invariance.* Every concept is a ratio of amplitudes within one lead, or a duration. Concepts are therefore invariant to any per-lead transformation x → a·x + b with a > 0, which makes them identical on z-scored (Melo) and mV (HUCA) data.
- *Sub-sample positions.* Values at fractional positions are obtained by linear interpolation, so gradients flow through the fiducial positions.

**Soft J point.** The QRS energy is computed from the right precordial leads, with each first difference scaled by the standard deviation σ~l~ of its lead:

$$
e_t = \sum_{l \in \{V1,V2,V3\}} \sqrt{\left(\Delta x_{l,t}/\sigma_l\right)^2 + \epsilon^2}, \qquad \hat e_t = e_t \,/\, \operatorname{softmax}_{|t - t_0| \le 6}\, e_t
$$

Here t₀ = 28 is the anchor, and the soft maximum is a scaled log-sum-exp. The J point is the soft *first* time after t₀ at which the normalized energy falls below a threshold θ:

$$
g_t = \sigma\!\left(\frac{\theta - \hat e_t}{\tau}\right), \qquad w_t = g_t \prod_{s<t} (1 - g_s), \qquad J = \sum_t w_t\, (t+1) + \Big(1 - \sum_t w_t\Big) (t_{\max}+1) + 2\tanh(\delta_J)
$$

Because ê~t~ is the difference between samples t and t + 1, a crossing at t places the J point at t + 1. The search runs over t ∈ \[t₀ + 1, t₀ + 14\]; τ = 0.05; θ ∈ \[0.03, 0.5\] is initialized at 0.15; and δ~J~ is a learnable shift of up to ±20 ms. The survival product makes J a differentiable analogue of "the first sample below threshold".

**Baseline and take-off point.**

- *Isoelectric baseline:* b~l~ is a Gaussian-weighted mean (σ = 15 ms) centred at a learnable offset of −150 to −50 ms before the QRS peak (initialized at −80 ms).
- *Take-off point:* because the visible take-off of a Brugada ST segment often lies after the global J point, each right precordial lead gets its own take-off J~l~. It is the soft-argmax of x~l~ − b~l~ over \[J − 10 ms, J + 50 ms\] (2.5 ms grid, temperature 0.02σ~l~).
- *Reference amplitude:* pp~l~, a soft peak-to-peak amplitude of the QRS complex.

**Concepts.** For each of V1, V2 and V3 (Table 2): x~J~ = x~l~(J~l~) − b~l~, x~80~ = x~l~(J~l~ + Δ) − b~l~ with a learnable Δ ∈ \[50, 110\] ms (initialized at 80 ms), and

$$
\text{J\_elev} = \frac{x_J}{pp}, \quad \text{ST80} = \frac{x_{80}}{pp}, \quad \text{Corrado} = \frac{x_J - x_{80}}{|x_J| + |x_{80}| + 0.05\,pp}
$$

The remaining six concepts per lead are:

- *slope* and *curv60:* the first- and second-order coefficients of quadratic fits on \[J~l~, J~l~ + 120 ms\] and \[J~l~, J~l~ + 60 ms\].
- *beta\_rel:* the angle between the upstroke on \[J~l~ − 30, J~l~\] ms and the downstroke on \[J~l~, J~l~ + 30\] ms, measured in amplitude/pp per 100 ms.
- *r\_prime:* a soft maximum over the last 60 ms of the QRS complex.
- *tri\_w:* the width of the r′–ST complex at a learnable fraction h ∈ \[0.05, 0.8\] of its peak (initialized at 0.5).
- *T:* the mean deviation 120–240 ms after J~l~.

All amplitudes are divided by pp~l~. A 28th, global concept, QRS\_w, is the soft width (ms) of the normalized energy above 0.3 within −80 to +150 ms.

The five learnable fiducial parameters (θ, δ~J~, baseline offset, Δ, h) are bounded through sigmoids. A variant that computes the energy from all nine leads was used as a comparator in the J-point validation (Section 2.7).

**Table 2. Concepts.**

| Concept | Clinical counterpart | Measured on |
| --- | --- | --- |
| J\_elev | J-point / ST take-off elevation | V1, V2, V3 |
| ST80 | ST elevation about 80 ms after take-off | V1, V2, V3 |
| Corrado | Corrado index (ST\_J / ST\_80); > 0 means a descending (coved) ST segment | V1, V2, V3 |
| slope, curv60 | ST slope and early convexity (coved versus saddle-back) | V1, V2, V3 |
| beta\_rel | β angle of the r′ wave | V1, V2, V3 |
| r\_prime | Terminal r′ amplitude | V1, V2, V3 |
| tri\_w | Base width of the r′–ST triangle | V1, V2, V3 |
| T | T-wave polarity | V1, V2, V3 |
| QRS\_w | QRS duration (energy-based) | V1–V3 energy |

## 2.6 Concept bottleneck models

A concept bottleneck model (CBM) standardizes the 28 concepts and applies a linear head whose weights carry clinically interpretable signs:

$$
p(y=1 \mid x) = \sigma\!\Big(\sum_k s_k\, \operatorname{softplus}(v_k)\, \tilde c_k(x) + b\Big), \qquad s_k \in \{-1, 0, +1\}
$$

**Sign gate.** The signs s~k~ are fixed before training, from the concepts at their initial (clinical) fiducial settings:

- s~k~ = +1 if the concept's AUROC exceeds 0.5 + 0.03 in *every* training domain;
- s~k~ = −1 if it is below 0.5 − 0.03 in every domain;
- s~k~ = 0 otherwise, which removes the concept.

A concept therefore enters the model only if it points the same way at all training sites, and its learned weight cannot flip sign to exploit a site-specific correlation.

**Normalization.** In the default model (CBM) concepts are standardized with the mean and standard deviation of the training pool, frozen at test time. The domain-adaptive variant (CBM-Ada) standardizes each training domain with its own statistics. At test time it uses the statistics of the unlabeled target batch, a form of test-time adaptation at the concept level.

**Training.**

- *Loss:* the average over training domains of the class-weighted binary cross-entropy, plus an L1 penalty (10⁻³) on the weights.
- *Optimization:* Adam (learning rate 0.02), full batch, at most 300 epochs, early stopping after 40 epochs without improvement in validation AUROC, random shifts of ±1 sample.
- *Ensemble:* two seeds, with probabilities averaged.
- *Fiducials:* the five fiducial parameters are learned jointly with the head.

**Variants.** CBM uses the 28 concepts above. CBM-noQRS removes the duration concept (s = 0). CBM-sd15 replaces QRS\_w with QRS\_sd15, a duration concept designed to be robust to device bandwidth.

QRS\_w is a soft threshold width of the derivative energy, so it depends on the high-frequency content that differs between acquisition devices. QRS\_sd15 is instead the RMS duration (square root of the second central moment) of the derivative energy envelope e(t) of V1–V3, computed within −80 to +140 ms after a 15 Hz low-pass filter. The envelope is normalized to its peak, and a floor of 0.1 is subtracted with a soft rectifier before the moments are taken:

$$
\mathrm{QRS}_{sd15} = \sqrt{\frac{\sum_t (t-\mu)^2\, e(t)}{\sum_t e(t)}}, \qquad \mu = \frac{\sum_t t\, e(t)}{\sum_t e(t)}
$$

Convolution with the filter adds the filter's own variance to the variance of the envelope. True duration differences are therefore preserved additively, while differences in device bandwidth above 15 Hz are removed.

QRS\_sd15 was chosen from eight candidate duration definitions by a rule that uses no labels. The rule required monotone tracking of QRS widening in phantoms (Spearman ρ ≥ 0.95, Section 2.7). Among the candidates that passed, it picked the one least sensitive to simulated device perturbations of the negative class: 30 Hz and 20 Hz low-pass filtering and 3-point smoothing, measured as the absolute standardized mean difference (|SMD|).

**Hybrid.** Hybrid models average the predicted probabilities of the V1–V3 CNN and a CBM (Hybrid = CNN-V1V3 + CBM; Hybrid-Ada = CNN-V1V3 + CBM-Ada). Averaging probabilities rather than ranks keeps the output on a probability scale, so that a validation threshold remains meaningful.

## 2.7 Physician-free validation of the concept layer

No clinician was available to annotate the BrS cohorts. We therefore validated the concept layer in two ways that need no new annotation: synthetic phantoms with known ground truth, and an external database annotated by cardiologists.

**Synthetic phantoms.**

- *Generation:* beats synthesized at 1000 Hz with a P wave, an rS complex with a terminal r′ in V1–V3 and a J point 80 ms after the R peak. The ST segment was normal (linear decline, J elevation 0–0.08 mV), coved (cosine descent with a negative T wave, 0.15–0.6 mV) or saddle-back (an early dip and rise with a positive T wave, 0.1–0.4 mV); r′ amplitude ranged over 0–0.5 mV.
- *Perturbation:* random per-lead gains (0.3–3), DC offsets, white noise and 0.3 Hz baseline wander were added, and the beats were downsampled to 100 Hz and passed through the alignment and z-scoring of Section 2.2.
- *Noise levels:* clean (noise SD 0.005); noisy (0.03, wander 0.15); and noisy with random alignment jitter of ±20 ms.
- *Checks:* (i) affine invariance, comparing concepts on raw and z-scored inputs; (ii) correctness of gradients against finite differences; (iii) recovery of the true J elevation and ST80 (Spearman correlation); (iv) discrimination of coved from saddle-back shapes (AUROC).

**LUDB J-point agreement.**

- *Processing:* the LUDB recordings went through the same pipeline as HUCA (band-pass filter, downsampling to 100 Hz, median beat, alignment, z-scoring). The concept layer never saw the annotations.
- *Error:* the soft J point of the median beat was mapped back to absolute time in each beat and compared with the cardiologist QRS-offset annotation of lead V2. The per-record error is the median over beats.
- *Reported statistics:* bias (mean error), SD, mean absolute error and the share of records within ±10 and ±20 ms, by conduction subgroup. As an acceptance criterion we used the CSE tolerance for QRS offset (SD ≤ 11.6 ms, i.e. 2σ~CSE~).
- *Variants compared:* energy from V1–V3 or from all nine leads; fiducials at their clinical initialization or as learned by a CBM trained on all three BrS cohorts; and a hard (non-soft) first-crossing threshold, to check that softening introduces no bias.

## 2.8 Concept-anchored threshold transfer (CATT)

**Problem.** A classifier f is trained on labelled source data, and its threshold t~src~ is chosen on a source validation set by Youden's J. At a new site only unlabelled ECGs are available. When the target score distribution shifts, t~src~ can land far from the target's balanced-accuracy-optimal threshold even if AUROC is preserved.

This is a shift of the scores, not of the prevalence alone. Under pure label shift the class-conditional score densities are unchanged, so the Youden threshold (where the likelihood ratio is about 1) is unchanged. Prevalence-correction methods therefore cannot repair it.

**Method.** CATT uses a second, structurally different model as a weak labeller of the target cohort: the concept model g.

1. Apply g with its own source threshold to obtain pseudo-labels ỹ = 1\[g(x) ≥ t~g~\] on the unlabelled target set.
2. Choose the threshold of f that maximizes Youden's J *against the pseudo-labels* on the same target set:

$$
\hat t_{\text{CATT}} = \arg\max_t \Big[\Pr\big(f(x) \ge t \mid \tilde y = 1\big) - \Pr\big(f(x) \ge t \mid \tilde y = 0\big)\Big]
$$

No target labels are used, and f itself is not modified. In all experiments g is the pre-specified CBM-sd15 ensemble.

**Why the miscalibration of g does not matter.** Let π be the target prevalence, α = P(ỹ = 1 | y = 0) and β = P(ỹ = 0 | y = 1) the error rates of the pseudo-labels, and q = P(ỹ = 1). If the errors of g are conditionally independent of the score of f given the true class (ỹ ⊥ f(x) | y), then each pseudo-class is a mixture of the two true classes. Writing λ₁ = P(y = 1 | ỹ = 1) and λ₀ = P(y = 1 | ỹ = 0),

$$
\tilde J(t) = (\lambda_1 - \lambda_0)\,\big[\mathrm{TPR}(t) - \mathrm{FPR}(t)\big] = \frac{\pi(1-\pi)\,(1-\alpha-\beta)}{q(1-q)}\; J(t)
$$

The Youden index measured on pseudo-labels is therefore a positive multiple of the true Youden index whenever α + β < 1, and both are maximized by the same threshold. This is the class-conditional label-noise invariance of balanced-error-optimal classifiers (Menon et al., 2015). The pseudo-labeller's own threshold error changes only α and β, so it shrinks the scaling factor without moving the maximizer.

In finite samples a small factor makes the empirical maximizer noisy. When the conditional-independence assumption is violated, the maximizer is biased towards the errors of g. These are the two quantities analysed in Section 2.9. The condition mirrors the view-independence assumption of co-training (Blum and Mitchell, 1998). A physiological measurement model is a natural second view of the ECG for a raw-signal CNN.

**Label-free comparators.** All use only source labels and unlabelled target data:

**Table 3.** Label-free threshold-transfer comparators. All methods are applied to the same trained target model.

| Method | Family | Description |
| --- | --- | --- |
| SRC | — | Source Youden threshold |
| EM-prior | Prior shift (Saerens et al., 2002) | Platt calibration on source validation; EM estimate of the target prior π̂; threshold the prior-adjusted posterior at π̂ (balanced-error optimal) |
| BBSE-Q | Prior shift (Lipton et al., 2018) | π̂ from the source confusion matrix at t~src~; threshold at the (1 − π̂) quantile of target scores |
| MED | Score-distribution matching | Map target median and IQR of the scores onto the source validation median and IQR |
| LSM | Mixture model (our initial proposal) | Target logits modelled as a two-component Gaussian mixture with source class shapes and a shared location shift δ; threshold t~src~ + δ |
| AdaBN | Test-time adaptation (Li et al., 2016; Schneider et al., 2020) | Batch-normalization statistics re-estimated on the target batch; source threshold |
| TENT | Test-time adaptation (Wang et al., 2021) | AdaBN plus entropy minimization of the batch-normalization affine parameters (Adam, learning rate 10⁻³, 10 full-batch steps, fixed a priori); source threshold |
| ORACLE | Upper bound | Youden threshold computed with target labels |

**Additional comparators.** To cover the strongest alternatives in the literature, we added five more:

- *UPA* (unsupervised prediction alignment; Roschewitz et al., 2023). The target score distribution is mapped onto the source-validation distribution by cumulative-distribution matching, and the source threshold is then applied. Equivalently, the threshold is the quantile of the target scores at F\_src(t~src~). UPA assumes that prevalence is unchanged between sites.
- *GMM-2.* A two-component Gaussian mixture is fitted to the target logits, and the threshold is placed where the two component densities cross (the balanced-error point if the mixture is correct).
- *Otsu.* Otsu's between-class-variance threshold on the target logits.
- *DANN* (Ganin and Lempitsky, 2015). Training-time adversarial feature alignment, with a gradient-reversal layer, a domain discriminator on the 64-dimensional CNN features and the standard λ schedule.
- *Deep CORAL* (Sun and Saenko, 2016). Training-time covariance alignment of the same features, with weight 1.

DANN and Deep CORAL used the unlabelled target ECGs during training. They kept the architecture, augmentation and early stopping on source-validation AUROC of CNN-V1V3, and their hyperparameters were fixed from the original papers without tuning. They were evaluated with their own source Youden threshold, and with CATT applied on top. We also tested three variants of CATT: soft pseudo-labels, confident pseudo-labels only, and a combined CBM-sd15 + ST\_V2 labeller.

AdaBN and TENT were recomputed for every target batch, so their dependence on batch composition (prevalence) is part of the evaluation.

**Development protocol.** To avoid tuning the method on the evaluation data, all design decisions about threshold transfer were made on two development settings only, HUCA → Claris and HUCA → Mortara. These decisions were abandoning LSM, and choosing plain CATT over a variant whose pseudo-labeller threshold is first shifted by LSM. The remaining seven settings are reported as held-out. The main target model is CNN-V1V3, and results are also reported for the hybrids.

## 2.9 Pseudo-labeller ablation and mechanism analysis

The identity in Section 2.8 makes two testable predictions about which pseudo-labeller should work best:

- It should have a large separation factor 1 − α − β.
- Its errors should be close to conditionally independent of the target model's score.

Neither prediction says that the pseudo-labeller needs a high AUROC. To test this, we kept the target model fixed (CNN-V1V3) and ran CATT with seven pseudo-labellers. Each was trained on the same source data and thresholded at its own source-validation Youden point.

**Table 4.** Pseudo-labellers compared in the ablation. CNNalt is the strongest possible control for view dependence: it is a near-copy of the target model.

| Group | Labeller | Description |
| --- | --- | --- |
| Concept | CBM-sd15 | Pre-specified default (Section 2.6) |
| Concept | CBM | Main concept model with QRS\_w |
| Concept | ST\_V2 | Single hand-crafted feature (ST-segment level in V2), no learning |
| Black box | LogReg9 | L2-regularized logistic regression on the raw 9-lead median beat |
| Black box | CNN9 | 9-lead CNN (Section 2.4) |
| Black box | ResNet | 1D ResNet on V1–V3 |
| Black box | CNNalt | Same architecture, inputs and data as the target CNN-V1V3, but initialization seeds 3–5 instead of 0–2 |

**Mechanism variables.** For every setting × validation seed × labeller we computed four quantities on the full target cohort, using the true labels for analysis only:

- α and β, the pseudo-label error rates.
- The separation factor 1 − α − β.
- The within-class dependence cond\_rho: the Spearman correlation between the labeller's score and the target model's score, computed separately in each true class and averaged over the two classes.
- The labeller's own AUROC.

The outcome was the gap between CATT and ORACLE balanced accuracy at natural prevalence.

**Analysis.** We averaged over seeds, giving 63 labeller × setting points (7 × 9). We report:

- Spearman correlations of the gap with each mechanism variable.
- A standardized linear regression of the gap on the two theory-derived variables, cond\_rho and 1 − α − β.
- The same regression with labeller AUROC added as a third predictor, to check whether AUROC carries information beyond the two theory-derived variables.

The regressions are descriptive. The 63 points are not independent because they share settings, so p-values are given only for the marginal correlations.

**Simulation.** To test the mechanism under controlled conditions, we simulated target batches with prevalence 0.2.

- The target-model scores are Gaussian within each class, f | y ∼ N(μ\_y, σ\_y²), with μ\_0 = 0, μ\_1 = 1.6, σ~0~ = 1 and σ~1~ = 1.6. The unequal variances put the true Youden threshold away from the midpoint.
- The pseudo-labeller score g has within-class correlation ρ with f and a class separation of 2.5 SD.
- g is thresholded at its correct midpoint, or shifted by c ∈ {−0.75, −1.25, −1.75, −2.25} SD to mimic a transferred threshold that has drifted. This sweeps 1 − α − β from 0.79 to 0.16.

We compared ρ = 0 with ρ = 0.9 at the two cohort sizes of this study (n = 356 and n = 1154), with 300 replicates per cell. The outcome was the gap between the balanced accuracy at the CATT threshold and at the true Youden threshold, both computed analytically.

Two further scenarios are reported in the supplementary files: a grid over ρ, labeller accuracy and n, and a shared-artefact model in which a subgroup of negatives raises both scores.

## 2.10 Sensitivity analyses and implementation

Two weaknesses of the data could plausibly change the conclusions, so we tested each directly.

**Reconstructed Melo labels.** We simulated the possibility that the true labels of the Claris and Mortara cohorts differ from the reconstructed labels in a fraction p ∈ {2, 5, 10}% of patients. Labels were flipped uniformly at random, with 20 draws per level.

The trained models, source thresholds and CATT pseudo-labels were kept exactly as computed, because in practice they were derived from the reconstructed labels. Only the evaluation on the Melo targets used the perturbed labels. This is a conservative scenario: retraining on correct labels could only help the models. We evaluated the six settings with a Melo target; settings with a HUCA target are unaffected.

**Median-beat construction for HUCA.** The Melo median beats were computed by the acquisition devices at a high sampling rate, whereas the HUCA median beats were computed by our pipeline at 100 Hz. Imprecise beat alignment smears a median beat (wider QRS, lower peaks). This could create part of the HUCA–Melo shift through the pipeline itself.

We therefore rebuilt the HUCA median beats with a device-style procedure:

1. Filter as in Section 2.2, then upsample the signal to 500 Hz.
2. Detect QRS complexes on the 500 Hz energy envelope.
3. Align each beat to a median template by cross-correlation (±20 ms search, 2 ms step, two iterations).
4. Take the median at 500 Hz, then downsample to 100 Hz with anti-alias filtering.
5. Re-anchor and z-score as before.

We retrained CNN-V1V3, the 9-lead CNN and CBM-sd15 on the seven settings involving HUCA, with the same validation seeds as the main analysis. We then repeated the AUROC and threshold-transfer evaluation.

**Patient overlap between the Melo devices.** For each beat we computed its nearest neighbour (Pearson correlation over the nine-lead median beat) in the other device's cohort. We compared this with nearest neighbours within a cohort, with HUCA ↔ Claris pairs, and with the same HUCA patient processed by the two median-beat pipelines.

**HUCA subgroups.** In the settings with a HUCA target, we computed AUROC separately for BrS-positive patients with and without a Brugada-type baseline pattern, each against all negatives. We also computed false-positive rates at the SRC, CATT and oracle thresholds for negatives with and without that pattern, and repeated the evaluation without the seven records whose label value is 2.

**Uncertainty of threshold-transfer results.** We retrained CNN-V1V3 with the same seeds, reproducing the stored scores to within 6 × 10⁻⁷, and saved per-patient scores for every method. We then drew 1000 patient-level bootstrap samples of each target cohort, stratified by class. In every sample we re-estimated every threshold that depends on the target batch (CATT and its variants, MED, UPA, GMM-2, Otsu, oracle); AdaBN, TENT, DANN and Deep CORAL scores were taken from the full batch. We report percentile 95% intervals for the difference between CATT and each method, per setting (Holm-adjusted over settings) and pooled over the held-out settings. At 5% prevalence we report the distribution over 200 resampled batches per setting.

**Sample size.** We set each threshold on a random subset of n ∈ {20, 30, 50, 75, 100, 150, 200, 300} target ECGs (100 draws each) and evaluated it on the whole target cohort.

**Concept decomposition of the shift.** For the CBM retrained in each setting (first validation split), the change in mean logit between source-validation and target patients of the same class was split into per-concept terms w\_k(z̄\_k,target − z̄\_k,source), using the frozen source standardization. Target labels were used only to separate the classes in this analysis.

**Implementation.** All code is in Python 3, using PyTorch for the neural and concept models, and NumPy, SciPy and scikit-learn for signal processing and statistics. All experiments ran on CPU. The full pipeline will be released with the paper: harmonization, concept layer, models, phantoms, LUDB validation, threshold transfer and every analysis script.

Public datasets used:

- HUCA and LUDB from PhysioNet.
- The Melo cohorts from the authors' public release.

# 3. Results

## 3.1 The three cohorts are separable domains, and source thresholds fail across them

**Domain separability.** After harmonization, a linear classifier identified the source cohort of a median beat with 95.5% accuracy (three classes, cross-validated). The AUROC was 0.998 for HUCA versus Melo, and 0.98 for Claris versus Mortara, the two Melo devices. The settings in Section 2.3 are therefore genuine distribution shifts, not resamplings of one population.

**No patient overlap between the Melo devices.** Claris and Mortara come from the same centre, so the same patients might have been recorded on both devices, which would leak patients between source and target in the Claris ↔ Mortara settings. We found no sign of this.

For every Mortara beat we found its most similar Claris beat (Pearson correlation over nine leads × 75 samples). The highest such correlation was 0.87 (median 0.69). This is the same level as between HUCA and Claris (maximum 0.87), and well below two other reference levels:

- the typical nearest neighbour within a cohort (median 0.89–0.90);
- the same HUCA patient processed by two different median-beat pipelines (median 0.96; Section 2.10).

No mutual nearest-neighbour pair across the two devices exceeded r = 0.95.

**Within-domain performance.** Under 5-fold cross-validation inside each cohort, the 9-lead CNN reached AUROC 0.929 on HUCA, 0.893 on Claris and 0.810 on Mortara. Logistic regression reached 0.872, 0.872 and 0.825.

The untrained single feature ST\_V2 reached 0.829, 0.790 and 0.805. Because it is not learned, these values are also its AUROC in every cross-domain setting with that target.

**Cross-domain performance.** Across the nine out-of-domain settings, the 9-lead CNN averaged AUROC 0.809 and logistic regression 0.793 (Table 5). Neither exceeded the unlearned ST\_V2 feature (0.808 on average over the same targets).

The larger failure was the decision threshold. When moving between HUCA and the Melo cohorts, the Youden threshold chosen on source validation data was badly misplaced in a direction that depended on the pair:

- From Melo to HUCA, the CNN labelled almost every HUCA ECG positive (specificity 0.15–0.22).
- From HUCA to Melo, it missed most Melo positives (sensitivity 0.27–0.34).

Ranking quality in these settings was preserved (AUROC 0.75–0.88). The collapse therefore concerns calibration of the threshold, not discrimination. Settings among the Melo cohorts or with HUCA in a pooled source were less affected.

**Table 5.** Nine-lead CNN baseline in the nine out-of-domain settings. Sensitivity and specificity are at the source-validation Youden threshold (single validation split).

| Setting | AUROC | Sensitivity | Specificity |
| --- | --- | --- | --- |
| HUCA → Claris | 0.751 | 0.27 | 0.96 |
| HUCA → Mortara | 0.785 | 0.34 | 0.96 |
| Claris → HUCA | 0.842 | 0.99 | 0.22 |
| Mortara → HUCA | 0.884 | 1.00 | 0.15 |
| Claris + Mortara → HUCA | 0.845 | 1.00 | 0.17 |
| Claris → Mortara | 0.838 | 0.66 | 0.90 |
| Mortara → Claris | 0.733 | 0.78 | 0.55 |
| HUCA + Mortara → Claris | 0.738 | 0.73 | 0.59 |
| HUCA + Claris → Mortara | 0.862 | 0.39 | 0.97 |
| **Mean** | **0.809** |  |  |

## 3.2 Restricting input to V1–V3 improves cross-domain discrimination

The diagnostic Brugada pattern is defined in the right precordial leads, whereas the other six leads mostly carry information about the site, device and population. Removing them improved out-of-domain AUROC for all three architectures (Table 6):

| Model | 9 leads | V1–V3 only | Settings improved |
| --- | --- | --- | --- |
| Logistic regression | 0.793 | 0.811 | 6 / 9 |
| CNN | 0.805 | 0.866 | 8 / 9 |
| 1D ResNet | 0.831 | 0.861 | 8 / 9 |

The gain was largest exactly where the 9-lead models were weakest. Examples:

- Mortara → Claris: CNN 0.733 → 0.865; ResNet 0.792 → 0.875.
- HUCA + Mortara → Claris: CNN 0.737 → 0.868; ResNet 0.761 → 0.871.
- Claris + Mortara → HUCA: CNN 0.814 → 0.909.

The only CNN setting that did not improve was Mortara → HUCA (0.884 → 0.864).

![](fig2_lead_restriction.png){width=16cm}

**Figure 2.** Out-of-domain AUROC with all nine leads (open grey) and with V1–V3 only (blue) for three architectures in the nine cross-site settings. Blue segments mark improvements, grey segments losses.

**Table 6.** Out-of-domain AUROC with all nine leads versus V1–V3 only. This is an independent training run from Table 5; differences in the shared 9-lead CNN columns reflect training stochasticity (at most 0.031).

| Setting | LogReg 9 | LogReg V1–V3 | CNN 9 | CNN V1–V3 | ResNet 9 | ResNet V1–V3 |
| --- | --- | --- | --- | --- | --- | --- |
| HUCA → Claris | 0.692 | 0.757 | 0.751 | 0.829 | 0.782 | 0.812 |
| HUCA → Mortara | 0.710 | 0.699 | 0.785 | 0.814 | 0.810 | 0.814 |
| Claris → HUCA | 0.870 | 0.856 | 0.842 | 0.876 | 0.868 | 0.873 |
| Claris → Mortara | 0.798 | 0.824 | 0.838 | 0.890 | 0.867 | 0.895 |
| Mortara → HUCA | 0.805 | 0.824 | 0.884 | 0.864 | 0.862 | 0.826 |
| Mortara → Claris | 0.789 | 0.847 | 0.733 | 0.865 | 0.792 | 0.875 |
| Claris + Mortara → HUCA | 0.867 | 0.859 | 0.814 | 0.909 | 0.884 | 0.894 |
| HUCA + Mortara → Claris | 0.795 | 0.814 | 0.737 | 0.868 | 0.761 | 0.871 |
| HUCA + Claris → Mortara | 0.808 | 0.821 | 0.860 | 0.879 | 0.853 | 0.886 |
| **Mean** | **0.793** | **0.811** | **0.805** | **0.866** | **0.831** | **0.861** |

The effect was stable across all later runs: CNN-V1V3 averaged 0.869–0.870 in the three-seed replications of Sections 3.4–3.6, against 0.816 for the 9-lead CNN. It also survived the device-style HUCA preprocessing (Section 3.9).

CNN-V1V3 was therefore used as the main black-box model and as the target model for threshold transfer.

## 3.3 The concept layer measures what it claims to measure

**Invariance and differentiability.** Concepts were invariant to per-lead affine transformations up to numerical precision: the maximum relative difference was 1.0 × 10⁻⁵, and 6 × 10⁻⁵ ms for QRS\_w. All gradients were finite. For the J-point threshold θ, the finite-difference gradient (−16.061) matched the autograd gradient (−16.062).

**Synthetic phantoms.** Phantom results are summarized in Table 7.

- *Graded ST elevation.* J\_elev tracked the true J-point elevation in V2 with Spearman ρ = 0.976 on clean beats, 0.936 after device perturbation, and 0.928 with additional alignment jitter. ST80 behaved similarly (0.98, 0.83, 0.84).
- *Morphology.* The Corrado concept separated coved from saddleback morphology with AUC 0.976 (clean), 0.971 (perturbed) and 0.884 (jitter). T-wave polarity in V2 separated them perfectly (AUC 1.0) under all conditions. The relative-slope concept β\_rel was the weakest (AUC 0.63–0.67), and is reported as such.
- *Abnormal versus normal.* J\_elev in V2 separated abnormal from normal phantoms with AUC 0.985–0.999.
- *Soft versus hard first crossing.* The soft first crossing gave the same tracking as a hard argmax (ρ 0.934 versus 0.930 in V2), so softening introduced no bias.

**Table 7.** Phantom validation of the concept layer.

| Test | Clean | Perturbed | Perturbed + jitter |
| --- | --- | --- | --- |
| J\_elev V2 vs true elevation (ρ) | 0.976 | 0.936 | 0.928 |
| ST80 V2 vs true elevation (ρ) | 0.980 | 0.825 | 0.835 |
| Corrado V2, coved vs saddle (AUC) | 0.976 | 0.971 | 0.884 |
| curv60 V2, coved vs saddle (AUC) | 0.896 | 0.938 | 0.863 |
| T\_V2, coved vs saddle (AUC) | 1.000 | 1.000 | 1.000 |
| J\_elev V2, abnormal vs normal (AUC) | 0.999 | 0.987 | 0.985 |

**J-point localization on LUDB.** On the 191 non-paced LUDB records, the soft J-point at its clinical initialization agreed with the cardiologist-annotated QRS offset with negligible bias (Table 8). Against the pre-specified CSE criterion (SD ≤ 11.6 ms), the nine-lead variant passed (SD 9.8 ms). The V1–V3 variant used in the concept layer passed in normal conduction (SD 7.7 ms) but not overall (SD 15.0 ms), because of conduction disorders (SD 23.6 ms).

- With energy from all nine leads: bias −0.6 ms, SD 9.8 ms, MAE 7.1 ms; 75% of records within ±10 ms and 95% within ±20 ms.
- With energy from V1–V3 only (the main layer): bias 0.0 ms, SD 15.0 ms.
- In records with normal conduction (n = 130), both variants were equally precise (SD 7.4 and 7.7 ms).
- The V1–V3 variant lost precision in conduction disorders (n = 61, SD 23.6 versus 13.6 ms), most visibly in left bundle-branch block (bias −14.8 ms). Here the terminal QRS forces in V1–V3 are atypical.
- The hard-threshold variants gave the same picture (bias +0.6 and +1.7 ms), confirming that the soft relaxation is unbiased.

**Table 8.** Soft J-point versus the annotated QRS offset in LUDB (non-paced records).

| Variant | n | Bias (ms) | SD (ms) | Within ±10 ms | Within ±20 ms |
| --- | --- | --- | --- | --- | --- |
| 9-lead energy, initial | 191 | −0.6 | 9.8 | 75% | 95% |
| 9-lead energy, hard | 191 | +0.6 | 9.8 | 80% | 96% |
| 9-lead energy, learned | 191 | +19.3 | 8.9 | 12% | 52% |
| V1–V3 energy, initial | 191 | 0.0 | 15.0 | 71% | 95% |
| V1–V3 energy, hard | 191 | +1.7 | 16.2 | 74% | 95% |
| V1–V3 energy, learned | 191 | +20.6 | 12.4 | 5% | 36% |
| V1–V3 energy, initial, normal conduction | 130 | +2.2 | 7.7 | 76% | 100% |
| V1–V3 energy, initial, conduction disorder | 61 | −4.5 | 23.6 | 61% | 84% |

**What the CBM learned.** When the fiducials were learned end-to-end by a CBM trained on the three BrS cohorts, the measurement point moved consistently later (values for the nine-lead-energy variant, with the V1–V3 variant in parentheses):

- θ decreased from 0.15 to 0.087 (0.088).
- δ~J~ converged to +13.5 ms (+13.8 ms).
- The baseline window moved from −80 to −100 ms (−102.5 ms).
- The ST offset moved from 80 to 90 ms (91 ms).

The learned J-point lay about 20 ms after the annotated QRS offset (bias +19.3 and +20.6 ms) with unchanged or better precision (SD 8.9 ms with nine-lead energy). The model thus chose to measure ST level slightly into the early ST segment rather than exactly at the end of the QRS complex. This is physiologically plausible for a disease defined by early ST elevation, and it was discovered without any J-point labels. The learned parameters were nearly identical across seeds (nine-lead variant: θ 0.0866 and 0.0864, δ~J~ 13.5 and 13.6 ms; V1–V3 variant: θ 0.0875 and 0.0876, δ~J~ 13.8 and 13.9 ms).

![](fig3_ludb_jpoint.png){width=16cm}

**Figure 3.** Physician-free validation of the soft J point on 191 non-paced LUDB records. (a) Error relative to the cardiologist-annotated QRS offset, by energy leads and fiducial state, separately for normal conduction and conduction disorders. Black bars are medians; the shaded band is ±10 ms. (b) Records with normal conduction: the soft J point at clinical initialization (grey) and after end-to-end learning by the CBM (blue), against the annotated QRS offset.

**QRS duration.** QRS\_w was only weakly correlated with the annotated QRS duration (Spearman 0.37). It should therefore be read as a QRS energy-width index, not a clinical QRS duration (see Section 3.5).

## 3.4 Concept bottleneck models: competitive ranking, more stable thresholds

Table 9 summarizes the main models over the nine out-of-domain settings (three validation seeds each).

**Table 9.** Out-of-domain performance. Balanced accuracy is at the source-validation Youden threshold.

| Model | Mean AUROC | Mean balanced accuracy | Worst-setting balanced accuracy |
| --- | --- | --- | --- |
| ST\_V2 (single feature) | 0.808 | 0.729 | 0.702 |
| CNN (9 leads) | 0.816 | 0.642 | 0.572 |
| CBM | 0.838 | 0.730 | 0.653 |
| CBM-Ada | 0.844 | 0.737 | 0.682 |
| CNN-V1V3 | 0.869 | 0.676 | 0.530 |
| Hybrid | 0.873 | 0.707 | 0.578 |
| Hybrid-Ada | 0.876 | 0.722 | 0.595 |

**Ranking.** The CBM, a sparse linear model on 28 physiological measurements, outperformed the 9-lead CNN by 0.022 AUROC and the single ST\_V2 feature by 0.030. It did not reach the V1–V3 CNN (0.838 versus 0.869). Averaging the CBM with CNN-V1V3 (Hybrid) gave 0.873. This was statistically indistinguishable from CNN-V1V3 in all nine settings (paired bootstrap with Holm correction: 0 wins, 9 ties, 0 losses). Hybrid was significantly better than the 9-lead CNN in 6 of 9 settings, with no losses.

The concept layer therefore does not improve discrimination over the best black box. Its value lies elsewhere: in the verifiable measurements of Section 3.3, and in the threshold behaviour described next.

**Thresholds.** At the source threshold, the concept models were markedly more stable than the CNNs. The CBM's mean balanced accuracy (0.730) exceeded that of CNN-V1V3 (0.676) despite its lower AUROC. Its worst setting (0.653) was 0.12 higher than CNN-V1V3's (0.530, Claris → HUCA, specificity 0.06).

The unlearned ST\_V2 had the best worst case of all (0.702). A fixed physical measurement transports its threshold better than a learned representation, and this is the observation that motivates CATT (Section 3.6).

**Domain-adaptive normalization depends on prevalence.** CBM-Ada slightly improved the mean AUROC (0.844) and the worst-case balanced accuracy (0.682). However, it normalizes concepts with statistics of the target batch, and those statistics depend on the batch's disease prevalence. In the prevalence stress test (run with the version of the concept layer that computes the J point and QRS\_w from nine-lead energy):

<!-- TODO: chạy lại với lớp V1–V3: python concepts_v3/stress_ada.py harmonized concepts_v3 v3 -> stress_summary_v3.csv, rồi cập nhật 3 gạch đầu dòng dưới đây -->

- The balanced accuracy of the CBM stayed at about 0.72 at every prevalence.
- The balanced accuracy of CBM-Ada rose from 0.719 at 5% prevalence to 0.740 at 35–50%.
- Its operating point drifted with prevalence: sensitivity/specificity 0.86/0.57 at 5%, and 0.61/0.87 at 70%.

Target-batch normalization is therefore unsuitable for screening populations, where prevalence is low and unknown. The fixed-normalization CBM was used as the default.

**Replacing the duration concept hurts.** Substituting the device-robust QRS\_sd15 for QRS\_w (CBM-sd15) reduced AUROC in all nine settings (0.812 versus 0.838; mean Δ −0.026, significant in 9/9 after Holm correction). Removing the duration concept altogether (CBM-noQRS) did the same (0.807; Δ −0.031, 9/9). The domain-adaptive version of CBM-sd15 reached 0.818. In hybrids, QRS\_sd15 also underperformed (0.857 versus 0.872; 5 losses, 4 ties). Section 3.5 explains why.

CBM-sd15 was nonetheless retained as the pre-specified CATT pseudo-labeller, because that role rewards robustness of the threshold rather than ranking (Section 3.8).

**What the CBM learned is stable and clinically coherent.** We retrained the CBM in each of the nine settings (first validation split). This reproduced the scores of the main analysis to within 6 × 10⁻⁷. Seventeen of the 28 concepts were active with the same sign in all nine settings. Five more never changed sign but were removed by the sign gate in some settings. Six changed sign between settings because the gate is recomputed on each training pool (curv60\_V2, curv60\_V3, ST80\_V2, tri\_w\_V2, tri\_w\_V3, r\_prime\_V3); curv60\_V2 had the fourth largest mean absolute weight (0.39). Table 9b lists the twelve consistent concepts with the largest weights.

The largest weights belonged to QRS energy width and J-point elevation in V2 (positive in 9/9 settings). Next came the terminal r′ amplitude in V2 and J-point elevation in V1 (positive), and ST slope, T-wave polarity in V2/V3 and the relative β-angle concept (negative). Together these describe a coved type-1 morphology: an elevated J point, a downsloping ST segment and an inverted T wave.

**Table 9b.** The most heavily weighted CBM concepts among those active with the same sign in all nine settings. Weights are on standardized concepts; the sign is fixed by the sign gate before training.

| Concept | Sign gate (+ / − out of 9) | Mean weight |
| --- | --- | --- |
| QRS\_w | 9 / 0 | +0.76 |
| J\_elev\_V2 | 9 / 0 | +0.44 |
| beta\_rel\_V1 | 0 / 9 | −0.41 |
| r\_prime\_V2 | 9 / 0 | +0.36 |
| beta\_rel\_V2 | 0 / 9 | −0.34 |
| J\_elev\_V1 | 9 / 0 | +0.32 |
| slope\_V1 | 0 / 9 | −0.30 |
| T\_V3 | 0 / 9 | −0.29 |
| slope\_V2 | 0 / 9 | −0.28 |
| Corrado\_V2, Corrado\_V3 | 9 / 0 | +0.27 |
| T\_V2 | 0 / 9 | −0.27 |

**Concept models shift less, and the concepts explain the shift.** For each model we measured how far the mean logit of target negatives moved away from that of source-validation negatives. With the source threshold fixed, this movement controls the specificity collapse. To compare models with different logit scales, we expressed it as a fraction of the model's own source class separation (the distance between the mean logits of source-validation positives and negatives).

In the three settings with a HUCA target, the negatives moved:

<!-- Nguồn số liệu: analysis_v8/logit_shift.py (logit_shift_per_setting.csv); đã chạy lại 15/10/2026, khớp. -->

- CNN-V1V3: 0.93–1.06 separations, i.e. almost exactly onto the source positives;
- CBM-sd15: 0.21–0.32 separations;
- CBM: 0.27–0.55 separations.

In the HUCA → Melo direction, the CNN negatives moved the other way (−0.41 to −0.43 separations).

Because the CBM is linear in standardized concepts, its shift splits exactly into one term per concept:

$$
\Delta\bar{\ell} = \sum_k w_k \left(\bar z_k^{\text{target}} - \bar z_k^{\text{source}}\right)
$$

The terms point to two concept families:

- QRS energy width was among the top contributors in every HUCA ↔ Melo setting, for both classes. It was higher in HUCA, which pushes HUCA logits up and Melo logits down (for example Mortara → HUCA: +1.65 for negatives and +1.72 for positives).
- ST level in V2 (J\_elev\_V2, ST80\_V2) was the other main contributor for negatives in Claris → HUCA (+0.56) and Mortara → HUCA (+2.51). In Claris + Mortara → HUCA it contributed only +0.28, and QRS energy width dominated.

The 23 HUCA negatives with a Brugada-type baseline pattern did not cause the shift. Removing them changed the logit shift of the negatives by at most 0.11 logit units (2–3% of the shift for CNN-V1V3, 4–15% for CBM-sd15).

## 3.5 The QRS width concept carries device-dependent diagnostic information

In isolation, QRS\_sd15 was better than QRS\_w on every criterion we measured (Table 10):

- It tracked phantom QRS widening more faithfully (ρ 0.98 versus 0.88).
- It was far less sensitive to simulated bandwidth changes (|SMD| 0.00 versus 0.47 under 30 Hz low-pass; 0.43 versus 2.0 under 20 Hz).
- It shifted less between real cohorts.
- It was a slightly better standalone discriminator in every cohort.

A third pre-specified candidate, QRS\_rel, was also robust but carried almost no diagnostic information (AUC 0.52–0.55). QRS\_rel is the log ratio of the QRS energy width in V1–V3 to that in V4–V6, motivated by the hypothesis of localized right-precordial conduction delay. This was the first sign that BrS widens QRS energy globally rather than only in the right precordial leads (see below).

**Table 10.** Duration concepts evaluated in isolation. SMD is computed in the negative class.

| Concept | Phantom widening (ρ) | \|SMD\|, 30 Hz low-pass | \|SMD\|, 20 Hz low-pass | Real shift C→M / H→C / H→M (SMD) | Standalone AUC H / C / M |
| --- | --- | --- | --- | --- | --- |
| QRS\_w | 0.88 | 0.47 | 2.00 | −0.48 / −0.93 / −1.41 | 0.71 / 0.66 / 0.71 |
| QRS\_sd15 | 0.98 | 0.00 | 0.43 | −0.34 / −0.46 / −0.82 | 0.74 / 0.70 / 0.77 |

Yet QRS\_sd15 made the CBM worse (Section 3.4). The reason is redundancy versus unique information:

- **QRS\_sd15 is redundant.** It correlated strongly with r\_prime\_V1 (ρ 0.59–0.71). After regressing out the other 27 concepts, its residual had no discriminative value (AUC 0.49–0.55).
- **QRS\_w carries unique information.** Its maximum correlation with any other concept was 0.26–0.36, and its residual retained AUC 0.58–0.60. Adding it to a logistic regression on the fixed concepts increased cross-domain AUROC by 0.014–0.058 in all six single-source pairs.

The unique information in QRS\_w is spectral. We had hypothesised that BrS shows a fragmented, high-frequency QRS; the data showed the opposite. ECGs with BrS had *less* energy above 15 Hz in the QRS-to-early-ST window (AUC 0.21–0.37 for the high-frequency energy ratio, i.e. 0.63–0.79 in the inverse direction). QRS\_w correlated negatively with this ratio (ρ ≈ −0.5). BrS also widened QRS energy in all leads, not only in V1–V3 (standalone AUC of the corresponding width in V4–V6 0.61–0.72).

This spectral property is intrinsically bandwidth-dependent. In the negative class, the high-frequency ratio was 0.18 in HUCA and 0.25 in the Melo cohorts.

Making the duration concept robust to devices therefore removes exactly the information that makes it useful. At the level of a single concept, the diagnostic and the device-dependent components cannot be separated. Because the analysis window includes the early ST segment, part of this signal may reflect ST elevation rather than QRS morphology.

We draw a practical conclusion: keep the informative but device-sensitive concept for ranking, and correct the resulting threshold shift at deployment time. CATT does this using an unlabelled target batch.

## 3.6 CATT repairs the transferred threshold without target labels

**The failure.** With three validation seeds, CNN-V1V3 kept a high AUROC in every setting (0.826–0.903). At the source threshold, however, its balanced accuracy collapsed to 0.55–0.59 in the five settings that cross between HUCA and Melo, while the oracle threshold would have given 0.76–0.84 (Table 11).

The errors had opposite signs in the two directions:

- From HUCA to Melo, the model was too conservative (HUCA → Claris: sensitivity 0.19, specificity 1.00).
- From Melo to HUCA, it was too liberal (Claris → HUCA and Claris + Mortara → HUCA: specificity 0.10).

**CATT.** Thresholding CNN-V1V3 at the Youden point computed against CBM-sd15 pseudo-labels restored balanced accuracy to 0.74–0.80 in these five settings, within 0.01–0.04 of the oracle. In the four settings where the source threshold was already adequate, CATT changed balanced accuracy by at most ±0.015. It therefore did not harm settings that did not need repair.

**Table 11.** Balanced accuracy of CNN-V1V3 at natural prevalence, by threshold-transfer method. For the broken settings, SRC sensitivity / specificity is shown in parentheses. dev = development setting.

| Setting | Split | SRC | LSM | CATT | ORACLE |
| --- | --- | --- | --- | --- | --- |
| HUCA → Claris | dev | 0.590 (0.19 / 0.99) | 0.686 | 0.770 | 0.775 |
| HUCA → Mortara | dev | 0.571 (0.16 / 0.98) | 0.680 | 0.741 | 0.757 |
| Claris → HUCA | held-out | 0.549 (1.00 / 0.10) | 0.733 | 0.772 | 0.809 |
| Claris → Mortara | held-out | 0.829 | 0.832 | 0.826 | 0.846 |
| Mortara → HUCA | held-out | 0.575 (0.96 / 0.19) | 0.701 | 0.791 | 0.803 |
| Mortara → Claris | held-out | 0.806 | 0.808 | 0.799 | 0.812 |
| Claris + Mortara → HUCA | held-out | 0.548 (1.00 / 0.10) | 0.715 | 0.797 | 0.837 |
| HUCA + Mortara → Claris | held-out | 0.793 | 0.764 | 0.787 | 0.802 |
| HUCA + Claris → Mortara | held-out | 0.808 | 0.781 | 0.823 | 0.830 |
| **Held-out mean** |  | **0.701** | **0.762** | **0.799** | **0.820** |
| **Held-out worst** |  | **0.548** | **0.701** | **0.772** | **0.802** |

**Low prevalence.** The held-out gain persisted when target batches were resampled to screening-like prevalence. CATT reached 0.794, 0.802 and 0.803 at 5%, 10% and 20% prevalence (10 resampled batches per setting; 0.790 at 5% over the 200 batches of Section 3.7), against 0.70 for SRC and 0.82–0.84 for the oracle.

On the held-out settings, CATT improved over SRC in 3–4 of 7 settings and worsened none by more than 0.01. The Wilcoxon signed-rank test reached p = 0.031 at 10% and 20% prevalence. At 5% and natural prevalence the p-values were 0.16 and 0.22, because only three or four settings had a threshold that needed repair; the remaining near-zero differences carry little signal. We therefore emphasize the size and consistency of the effect in the broken settings, not the p-value.

**How many unlabelled target ECGs are needed?** In deployment the threshold must be set from the first ECGs a site acquires. We therefore chose the CATT threshold on a random subset of n target ECGs (100 draws per n, natural prevalence) and applied it to the whole target cohort. Held-out balanced accuracy, mean (10th percentile over draws), was:

| n | 20 | 50 | 100 | 200 | all |
| --- | --- | --- | --- | --- | --- |
| CATT | 0.766 (0.725) | 0.782 (0.753) | 0.789 (0.771) | 0.795 (0.783) | 0.799 |

The source threshold gave 0.701 regardless of n. CATT therefore needs only 50–100 unlabelled ECGs to recover most of the gain; at very small n (20–30) UPA was as good (Figure 6b).

**Why the mixture-model approach failed.** LSM, our initial proposal, assumes that each class's target score distribution is a shifted copy of its source shape. It recovered only part of the gap (held-out 0.762, worst 0.701) and was abandoned on the development settings. Under domain shift the CNN's scores did not shift rigidly; they compressed. In the broken settings, the separation between class means fell from 1.7 logit units on source validation to as little as 0.36 on the target, which violates the shape assumption.

A variant that first shifted the pseudo-labeller's threshold with LSM (CATT-LSM) was no better on development (0.750 versus 0.756) or held-out (0.792 versus 0.799) data, and was not pursued.

**Other target models.** CATT also improved the hybrids. For Hybrid, held-out balanced accuracy went from 0.716 to 0.789 at natural prevalence (oracle 0.817), and the worst setting from 0.581 to 0.754. The gains were smaller than for CNN-V1V3 because averaging with the CBM had already partly stabilized the threshold.

Applied to the CBM itself, CATT gave +0.019. Applied to CBM-sd15 it is the identity by construction, since the pseudo-labeller and the target model coincide.

## 3.7 Comparison with label-free adaptation methods

Table 12 compares CATT with five families of label-free alternatives on the seven held-out settings. All were applied to the same CNN-V1V3 models:

- prior-shift correction (EM-prior, BBSE-Q);
- score-distribution matching (MED, UPA);
- unsupervised thresholding of the target scores (GMM-2, Otsu);
- test-time adaptation (AdaBN, TENT);
- training-time domain adaptation (DANN, Deep CORAL).

**Table 12.** Held-out settings (n = 7): mean and worst-setting balanced accuracy of CNN-V1V3. At 5% prevalence, values are averaged over 200 resampled target batches per setting. †From the run of Section 3.6 (same models, 10 batches at 5% prevalence).

| Method | Bal. acc., natural | Worst, natural | Bal. acc., 5% | Worst, 5% |
| --- | --- | --- | --- | --- |
| SRC | 0.701 | 0.548 | 0.703 | 0.548 |
| EM-prior† | 0.707 | 0.550 | 0.705 | 0.550 |
| BBSE-Q† | 0.680 | 0.502 | 0.647 | 0.502 |
| MED | 0.773 | 0.713 | 0.754 | 0.732 |
| UPA | 0.786 | 0.762 | 0.751 | 0.728 |
| GMM-2 | 0.785 | 0.751 | 0.774 | 0.686 |
| Otsu | 0.783 | 0.742 | 0.772 | 0.687 |
| AdaBN | 0.765 | 0.714 | 0.767 | 0.714 |
| TENT | 0.767 | 0.715 | 0.769 | 0.715 |
| DANN | 0.703 | 0.537 | 0.705 | 0.537 |
| Deep CORAL | 0.701 | 0.533 | 0.703 | 0.533 |
| **CATT** | **0.799** | **0.772** | **0.790** | **0.744** |
| CATT, soft pseudo-labels | 0.799 | 0.786 | 0.794 | 0.768 |
| DANN + CATT | 0.800 | 0.768 | 0.797 | 0.762 |
| ORACLE | 0.820 | 0.802 | 0.842 | 0.821 |

**Prior-shift correction does not help.** EM-prior and BBSE-Q are designed for label shift, and they cannot repair a shift of the class-conditional score distributions (Section 2.8). EM-prior left the threshold essentially unchanged (0.707), and BBSE-Q made it worse (0.680; worst setting 0.502). Both estimated the target prevalence badly: π̂ was 0.36–0.42 when the true prevalence was 0.05–0.20.

**Training-time adaptation does not move the operating point.** DANN and Deep CORAL used the unlabelled target ECGs during training, with model selection on source validation data only. They kept or slightly raised discrimination (held-out AUROC 0.880 and 0.876, versus 0.876 for CNN-V1V3). Their source threshold failed exactly as before (0.703 and 0.701; worst settings 0.537 and 0.533). Aligning feature distributions does not align the decision threshold. CATT applied on top of DANN worked as well as on the plain CNN (0.800).

**Test-time adaptation helps partially.** AdaBN and TENT recovered about half of the gap to the oracle (0.765–0.767). Their benefit came entirely from moving the operating point, since they did not change discrimination (AUROC 0.876 → 0.877). At 5% prevalence the target batch is dominated by negatives. The adapted network then stays too liberal: in the run of Section 3.6, its specificity was 0.56–0.57, against 0.75 for CATT at similar sensitivity (0.83).

**Score matching and unsupervised thresholding are the strongest alternatives.** UPA matches the whole score distribution of the target batch to that of the source validation set (Roschewitz et al., 2023). GMM-2 fits a two-component Gaussian mixture to the target logits and thresholds where the two component densities cross. At natural prevalence, both came within 0.013–0.015 of CATT (0.786 and 0.785). Their weaknesses appeared under stress:

- UPA assumes that prevalence is unchanged between sites. At 5% prevalence it fell to 0.751, as did MED (0.754).
- GMM-2 and Otsu kept their means at 5% prevalence (0.774 and 0.772), but their worst setting dropped to 0.686–0.687. A mixture of two Gaussians becomes unstable when one class is rare.

CATT was the only method that stayed high on both criteria at both prevalences (0.799 / 0.772 natural; 0.790 / 0.744 at 5%).

**Statistical comparison.** We used a patient-level stratified bootstrap (1000 resamples). In each resample, every batch-dependent threshold was re-estimated. Table 12b gives the pooled held-out differences.

**Table 12b.** Pooled difference in held-out balanced accuracy between CATT and each alternative. At 5% prevalence the interval is over 200 resampled target batches per setting.

| CATT minus | Natural prevalence, Δ \[95% CI\] | 5% prevalence, Δ \[95% interval over batches\] |
| --- | --- | --- |
| SRC | +0.098 \[0.084, 0.109\] | +0.088 \[0.068, 0.107\] |
| MED | +0.026 \[0.013, 0.037\] | +0.036 \[0.017, 0.051\] |
| UPA | +0.013 \[0.003, 0.022\] | +0.039 \[0.020, 0.055\] |
| GMM-2 | +0.015 \[0.002, 0.027\] | +0.016 \[−0.001, 0.030\] |
| Otsu | +0.016 \[0.002, 0.027\] | +0.018 \[0.002, 0.031\] |
| AdaBN | +0.034 \[0.021, 0.042\] | +0.024 \[0.006, 0.042\] |
| TENT | +0.032 \[0.019, 0.041\] | +0.022 \[0.004, 0.039\] |
| DANN | +0.096 \[0.081, 0.107\] | +0.085 \[0.063, 0.103\] |
| Deep CORAL | +0.098 \[0.083, 0.109\] | +0.088 \[0.064, 0.109\] |

Per setting, CATT exceeded SRC by 0.17–0.25 in each of the five settings where the source threshold fails (Holm-adjusted bootstrap p < 0.001). It differed by −0.007 to +0.014 in the other four settings, where no difference was significant. Against UPA, CATT was significantly better in three settings and significantly worse in none. Against GMM-2 no single-setting difference survived Holm correction. The advantage over the unsupervised-thresholding baselines is therefore small and consistent rather than large.

**Design choices of CATT do not matter.** Three variants gave results indistinguishable from the pre-specified hard pseudo-labels (all pooled |Δ| ≤ 0.005, all CIs including 0):

- soft pseudo-labels (each target ECG weighted by g's probability, re-centred on g's source threshold);
- confident pseudo-labels only (excluding a band of ±0.5 SD around g's threshold);
- a combined labeller (averaged standardized CBM-sd15 and ST\_V2 scores).

The soft variant had the best worst case (0.786 natural, 0.768 at 5%), but we report the pre-specified version as the main result.

**Development settings.** On HUCA → Claris and HUCA → Mortara, CATT reached 0.770 and 0.741, against 0.716 and 0.726 for UPA and 0.755 and 0.730 for GMM-2. Its margins on the held-out settings were of similar size, so the result is not an artefact of design choices made on the development pair.

![](fig4_catt.png){width=16cm}

**Figure 4.** Threshold transfer for CNN-V1V3. (a) Balanced accuracy per setting at natural prevalence with the source threshold (open grey), with CATT (blue) and with the oracle threshold (black tick). Blue segments mark the five settings where the source threshold fails; dev = development setting. (b) Mean held-out balanced accuracy (seven settings) of the label-free methods across target prevalences.

## 3.8 A good pseudo-labeller is a different view, not a more accurate one

Table 13 shows CATT with each of the seven pseudo-labellers, averaged over the nine settings. For reference, SRC averaged 0.674 and ORACLE 0.808 over the same settings at natural prevalence.

**Table 13.** CATT with different pseudo-labellers; the target model is CNN-V1V3 in every row. Means over nine settings at natural prevalence, except the "At 5%" column and the held-out column (seven settings). cond\_rho is the within-class Spearman correlation between the labeller's score and the target model's score.

| Labeller | Type | Labeller AUROC | α | β | 1 − α − β | cond\_rho | CATT bal. acc. | Worst | At 5% | Held-out |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CBM-sd15 | concept | 0.808 | 0.24 | 0.30 | **0.46** | **0.43** | **0.790** | **0.741** | **0.779** | **0.799** |
| CBM | concept | 0.835 | 0.27 | 0.27 | 0.46 | 0.47 | 0.783 | 0.719 | 0.769 | 0.793 |
| ST\_V2 | concept | 0.808 | 0.20 | 0.36 | 0.44 | 0.52 | 0.782 | 0.740 | 0.778 | 0.789 |
| LogReg9 | black box | 0.801 | 0.36 | 0.26 | 0.38 | 0.49 | 0.769 | 0.724 | 0.759 | 0.772 |
| CNN9 | black box | 0.812 | 0.34 | 0.39 | 0.27 | 0.57 | 0.748 | 0.651 | 0.742 | 0.754 |
| ResNet | black box | 0.861 | 0.31 | 0.31 | 0.38 | 0.87 | 0.710 | 0.570 | 0.714 | 0.739 |
| CNNalt | black box | **0.868** | 0.35 | 0.30 | 0.35 | 0.89 | 0.703 | 0.602 | 0.699 | 0.729 |

**Ranking of labellers.** The three concept-based labellers were the best, and they were close to one another (held-out 0.789–0.799). The four black-box labellers were worse, in the order predicted by their dependence on the target model.

The two most accurate labellers, ResNet and CNNalt (AUROC 0.861 and 0.868), gave the worst CATT results (0.710 and 0.703; worst setting 0.570 and 0.602). Their within-class correlation with CNN-V1V3 was 0.87–0.89, so they repeated its errors. CNNalt is the same network trained with different seeds, and it reproduced the target's threshold failure in the HUCA-target settings: α = 0.80–0.91, i.e. most HUCA negatives were pseudo-labelled positive.

On the held-out settings, CATT with CBM-sd15 was better than CATT with:

- CNN9: +0.048 at 5% prevalence (6 wins, 0 losses; p = 0.016) and +0.045 at natural prevalence (p = 0.078).
- ResNet: +0.052 at 5% and +0.060 at natural prevalence.
- CNNalt: +0.063 and +0.070.
- LogReg9: +0.021 and +0.027.

Against the other two concept labellers the differences were 0.004–0.017 and not significant.

**Mechanism.** Across the 63 labeller × setting points:

- The gap to the oracle increased with dependence on the target model (Spearman ρ = +0.33, p = 0.008).
- The gap decreased with the separation factor 1 − α − β (ρ = −0.62, p = 5 × 10⁻⁸).
- The two predictors were essentially uncorrelated with each other (ρ = −0.02).

A standardized linear model with only these two theory-derived variables explained most of the variance:

$$
\text{gap} = 0.026\,\tilde{\rho}_{\text{cond}} \;-\; 0.046\,\widetilde{(1-\alpha-\beta)}, \qquad R^2 = 0.815
$$

Here ρ\_cond is cond\_rho, and tildes denote standardized variables.

The labeller's own AUROC was *positively* correlated with the gap (ρ = +0.46, p = 2 × 10⁻⁴), and it kept a positive coefficient when added to the model (R² = 0.898). In other words, holding separation and dependence fixed, a more accurate labeller did not make CATT better. Among the labellers available here, higher accuracy came with greater similarity to the target network, and therefore with shared errors.

![](fig5_labeller_mechanism.png){width=16cm}

**Figure 5.** What makes a good CATT pseudo-labeller. Gap between CATT and oracle balanced accuracy against (a) the labeller's within-class correlation with the target CNN and (b) the separation factor 1 − α − β. Small dots are labeller × setting points; large dots are means over the nine settings.

**Simulation: when does dependence hurt?** The real-data mechanism analysis is observational, so we also simulated CATT under controlled conditions (Section 2.9):

- the target model's scores have unequal class variances, so the true Youden threshold is not at the midpoint;
- the pseudo-labeller's within-class correlation with the target model (ρ) is controlled;
- the pseudo-labeller's threshold is shifted to control α and β.

Two results stood out (Figure 6a):

- *Independent labeller (ρ = 0).* Even a badly miscalibrated labeller only added sampling noise. The gap to the oracle grew from 0.009 to 0.040 as 1 − α − β fell from 0.79 to 0.16 (n = 356), and it roughly halved when n tripled.
- *Strongly dependent labeller (ρ = 0.9).* Dependence was harmless while the labeller's threshold was correct (gap 0.004). Once the threshold was off, dependence produced a systematic bias: 0.03, 0.07, 0.11 and 0.15 as 1 − α − β fell to 0.67, 0.49, 0.30 and 0.16. This bias did not shrink with n (identical at n = 356 and n = 1154).

The harm therefore comes from the combination of the two conditions in Section 2.8. A labeller whose errors resemble the target model's, and whose own transferred threshold is wrong, drags the CATT threshold with it.

This is exactly the situation of the black-box labellers in the HUCA-target settings. There, CNNalt pseudo-labelled 80–91% of negatives as positive, and its CATT gap (0.15–0.23) matched the simulated bias. The concept labellers were less dependent on the target CNN, and their own thresholds also shifted far less (Section 3.4), so they avoided both conditions.

![](fig6_simulation_samplesize.png){width=16cm}

**Figure 6.** (a) Simulated gap between CATT and oracle balanced accuracy as the pseudo-labeller's threshold is moved (lower 1 − α − β), for an independent (blue) and a strongly dependent (orange) labeller, at the two cohort sizes of this study. (b) Held-out balanced accuracy when the threshold is set from n unlabelled target ECGs and applied to the whole cohort. The shading shows the 10th percentile of CATT over 100 draws.

These results support both conditions of the identity in Section 2.8. They indicate that the value of the concept layer for threshold transfer comes from being a structurally different view of the ECG, not from being a better classifier.

The single unlearned feature ST\_V2 was almost as good a labeller as the CBMs (held-out 0.789 versus 0.799; not significant). A strong, simple physiological measurement may therefore suffice as the anchor. The added value of the full CBM lies in interpretability and in the verifiable multi-concept measurements of Section 3.3, not in a measurably better threshold anchor.

## 3.9 Sensitivity analyses

**Reconstructed Melo labels.** Randomly corrupting 2%, 5% or 10% of the Melo evaluation labels lowered every absolute metric roughly in parallel, as label noise pulls all measured performance towards chance. No comparison changed (Table 14).

In the two development settings where the source threshold fails, the gain from CATT stayed large:

- HUCA → Claris: +0.17, +0.16 and +0.14 at 2%, 5% and 10% corruption.
- HUCA → Mortara: +0.16, +0.15 and +0.13.

In the other Melo-target settings it remained within ±0.01. CATT stayed 0.012–0.013 below the oracle at every noise level.

**Table 14.** Mean over the six settings with a Melo target (20 corruption draws per level). Models, source thresholds and pseudo-labels were left as trained on the reconstructed labels.

| Label corruption | AUROC (CNN-V1V3) | SRC | CATT | ORACLE | CATT − SRC |
| --- | --- | --- | --- | --- | --- |
| 0% (as reported) | 0.860 | 0.733 | 0.791 | 0.804 | +0.058 |
| 2% | 0.844 | 0.723 | 0.778 | 0.791 | +0.055 |
| 5% | 0.821 | 0.707 | 0.759 | 0.772 | +0.052 |
| 10% | 0.782 | 0.683 | 0.729 | 0.741 | +0.046 |

**Device-style median beats for HUCA.** The re-derived HUCA median beats were well aligned:

- The median within-record SD of beat-alignment lags was 0.57 ms.
- After compensating anchor shifts of one to three samples (in 26% of records), the median correlation with the original beats was 0.966.
- Re-running the original pipeline reproduced the dataset used in the paper exactly, ruling out version drift.

Retraining on the new HUCA beats changed none of the conclusions (Table 15). The HUCA–Melo threshold failure persisted: SRC balanced accuracy was 0.56–0.60 in the five crossing settings, against 0.55–0.59 before. CATT repaired it to the same degree, and the V1–V3 advantage was preserved (0.862 versus 0.819 for the 9-lead CNN).

The threshold shift between HUCA and Melo is therefore not an artefact of our median-beat construction. It must come from differences between the populations or acquisition chains themselves.

A difference in how disease was confirmed is unlikely to be the main cause. In both cohorts most positives lack a spontaneous pattern on the analysed ECG: 52 of 73 HUCA positives (71%) have no Brugada-type baseline pattern, and 83% of Melo positives were diagnosed only on ajmaline challenge.

The concept decomposition of Section 3.4 instead points to ST-segment level in V2 and QRS energy width, which are higher in HUCA ECGs of both classes. With three cohorts, population and acquisition effects cannot be separated further.

**HUCA subgroups and the undefined label.** We evaluated the three settings with a HUCA target by baseline-pattern subgroup (Table 14b).

- *Concealed BrS was still detected.* CNN-V1V3 separated positives without a Brugada-type baseline pattern (n = 52) from all negatives with AUROC 0.84–0.89. For positives with the pattern (n = 21) the AUROC was 0.88–0.93.
- *False positives were concentrated in phenocopies.* At the CATT threshold, 32–64% of the 23 negatives with a Brugada-type baseline pattern but no diagnosis were called positive, against 17–34% of the other negatives. Clinically, these are the ECGs a cardiologist would also flag.
- *The undefined label did not matter.* Excluding the seven records with label value 2 changed AUROC by at most 0.011 and CATT balanced accuracy by at most 0.008.

**Table 14b.** HUCA-target settings by baseline-pattern subgroup. FPR = false-positive rate. Means over three validation seeds.

| Target (CNN-V1V3) | AUROC, pattern-positive BrS | AUROC, concealed BrS | FPR at CATT, pattern-positive negatives | FPR at CATT, other negatives | CATT bal. acc. (excl. label 2) |
| --- | --- | --- | --- | --- | --- |
| Claris → HUCA | 0.884 | 0.861 | 0.64 | 0.34 | 0.772 (0.767) |
| Mortara → HUCA | 0.908 | 0.842 | 0.32 | 0.17 | 0.791 (0.784) |
| Claris + Mortara → HUCA | 0.930 | 0.886 | 0.49 | 0.25 | 0.797 (0.789) |

**Table 15.** Sensitivity to the HUCA median-beat construction (natural prevalence, three validation seeds). —, not computed: the 9-lead CNN was not retrained in the original-beat run.

| Metric (7 settings involving HUCA) | Original HUCA beats | Device-style HUCA beats |
| --- | --- | --- |
| AUROC, CNN-V1V3 | 0.860 | 0.862 |
| AUROC, CNN (9 leads) | — | 0.819 |
| AUROC, CBM-sd15 | 0.804 | 0.805 |
| Balanced accuracy, SRC | 0.633 | 0.640 |
| Balanced accuracy, CATT | 0.783 | 0.786 |
| Balanced accuracy, ORACLE | 0.802 | 0.805 |

**Table 16.** Per-setting balanced accuracy of CNN-V1V3 before and after CATT, under both HUCA preprocessing pipelines.


| Setting | Original: SRC → CATT | Device-style: SRC → CATT |
| --- | --- | --- |
| HUCA → Claris | 0.590 → 0.770 | 0.593 → 0.772 |
| HUCA → Mortara | 0.571 → 0.741 | 0.572 → 0.752 |
| Claris → HUCA | 0.549 → 0.772 | 0.557 → 0.771 |
| Mortara → HUCA | 0.575 → 0.791 | 0.599 → 0.787 |
| Claris + Mortara → HUCA | 0.548 → 0.797 | 0.555 → 0.806 |
| HUCA + Mortara → Claris | 0.793 → 0.787 | 0.787 → 0.798 |
| HUCA + Claris → Mortara | 0.808 → 0.823 | 0.816 → 0.816 |
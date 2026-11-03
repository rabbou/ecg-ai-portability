# Correcting an ECG model's positive predictive value at an external site with few local labels

## Abstract

**Background.** A site adopting an ECG model receives a positive predictive value (PPV) measured in the developer's population and adjusts it to its own prevalence by Bayes' theorem. This prevalence adjustment assumes that the sensitivity and specificity of a fixed threshold are transportable between populations. An earlier measurement found that a shift in the operating point violated it between hospitals and care settings. A site can instead update the model on local labels, but the number of labels such an update requires has not been measured.

**Methods.** Five updating methods, which state the PPV at an external site, were compared in 105 transfers, each a model calibrated in one population and applied to another, at Columbia, Beth Israel Deaconess and Chongqing. Two methods used no labels; three used label budgets of 25 to 400 ECGs, drawn by patient, with 200 draws per budget. The primary outcome was the absolute error between the stated PPV and the PPV observed in the unlabelled target patients. The proportion of draws in which a threshold refitted on the labels flagged every patient was made a co-primary outcome after the results were read.

**Results.** Without labels, the prevalence correction erred by a median of 13.1, 11.4 and 51.7 percentage points at the three sites. With 100 labels, which held a mean of 9.9, 13.2 and 14.8 cases, logistic recalibration erred by 3.1, 5.7 and 3.8 points. The refitted threshold flagged every patient in 79%, 84% and 71% of the draws in which it was placed, and 99%, 99% and 98% of cases. Its error was 2.6, 2.2 and 3.5 points over these draws, and 5.1, 4.3 and 5.5 points over the draws in which its threshold was finite. A threshold certified by exact binomial tests to hold the source PPV was found in 16.8%, 25.1% and 1.0% of draws; among these draws, its observed PPV fell below the source PPV in 13.2%, 5.7% and 100%.

**Conclusions.** With 100 local labels, recalibration stated the PPV within a few points at three external sites. A threshold refitted on 100 labels mostly flagged every patient, and its small error at that budget did not reflect a threshold that separates cases from non-cases. These results suggest that a refitted threshold needs more labels than recalibration, enough to place nine cases in the half used to fit it.

## 1. Introduction

A site adopting an ECG model receives a PPV measured in the developer's population. Standard practice is prevalence adjustment: the PPV is recomputed at the local prevalence by Bayes' theorem. The adjustment assumes that the sensitivity and specificity of the model's threshold are transportable, so that only the prevalence differs between the two populations. A change in the severity or presentation of cases, or in the non-cases, violates it.

In a retrospective measurement [1], the adjusted PPV did not reliably predict the PPV observed at other hospitals and care settings, because the operating point of a fixed threshold shifted between populations. A site can instead update the model on local labels, ECGs whose diagnosis the site has confirmed: it can recalibrate the model's probabilities, refit its threshold, or certify a threshold by a statistical test. A PPV held constant across sites requires a threshold that moves with the site, and a higher threshold trades sensitivity for PPV. The number of labels such an update requires, and the sensitivity it retains, have not been measured.

We compared five updating methods by the agreement between the PPV each states and the PPV observed at the external site, as a function of the number of local labels.

## 2. Methods

### 2.1 Sites and transfers

The unit of analysis is the transfer, from a source population to a target population. Three external sites form the primary analysis. At Columbia, the four EchoNext models, calibrated on inpatients, were applied to emergency and outpatient ECGs (76 transfers). At Beth Israel Deaconess, the models of the seven-source rotation were applied to MIMIC-IV-ECG (28 transfers), whose labels are the ECG machine's statements, so that the observed PPV there measures agreement with the machine rather than with a clinical reference standard. At Chongqing, the PTB-XL myocardial infarction model was applied to the hospital's ECGs (1 transfer). Shandong, the other rotation corpora, CODE-15%, the CODE test set and ZZU form a secondary reserve, reported in [`results/ppv_repairs_reserve.json`](results/ppv_repairs_reserve.json). Transfers in which a model is evaluated in its own source population serve as in-distribution controls.

### 2.2 Target PPV and updating methods

For each transfer, the target PPV v is the PPV attained in the source population at the threshold that gives 90% sensitivity there. It was fixed from the source scores alone, before any method was run, and the results file records the digest of the target table. The five updating methods were:

1. prevalence correction without labels, in which the target prevalence was estimated from the model's scores by expectation-maximisation, an iterative maximum-likelihood estimate of the class proportions [2], and the PPV at the source threshold was recomputed by Bayes' theorem;
2. logistic recalibration on n labels, which refits the intercept and slope of the model's logit to the labels, or the intercept alone when the rarer class has fewer than ten events; the stated PPV is the mean recalibrated probability among flagged patients;
3. threshold refitting on split labels, in which the threshold was set on one half of the n labels to give 90% sensitivity and the stated PPV was the PPV observed at that threshold in the other half, with its one-sided Clopper-Pearson bound, an exact binomial confidence bound;
4. threshold certification by Learn then Test [3], in which thresholds from a grid fixed by the source scores were tested in fixed sequence from the highest down, each by an exact binomial test of PPV ≤ v at δ = 0.1, and the lowest threshold rejected before the first non-rejection was retained; the stated PPV is v. The guarantee is marginal: over repeated labelled samples, the probability that a threshold is retained whose PPV is at or below v is at most δ. It does not bound that probability among the samples in which a threshold is retained;
5. weighted conformal selection without labels ([4], Algorithm 1 with deterministic pruning), which flags patients so that the false discovery rate, the expected proportion of false positives among flagged patients, stays below q = 1 − v; covariate-shift weights, the estimated ratio of target to source density of the model's score, adjust the source patients to the target population.

Methods 3 and 4 place thresholds by the conformal rule used at the source: among m cases, the threshold is the score of the case ranked ⌈(m + 1) × 0.9⌉ from the highest. When a half-sample holds fewer than nine cases, this rank exceeds m, the threshold is infinite and every patient is flagged. A threshold that flags every patient states and observes the prevalence, so its error is small whether or not it separates cases from non-cases.

### 2.3 Sampling and outcomes

Labels were sampled by patient at budgets of n = 25, 50, 100, 200 and 400, with 200 draws per budget. Target patients not sampled formed the evaluation set; methods 1 and 5 used the whole target. The protocol named as primary outcome the absolute error, in percentage points, between the stated PPV and the PPV observed in the evaluation set, and as secondary outcomes sensitivity and net benefit. Sensitivity in the target was averaged over the draws in which a method returned a threshold, then over the transfers of a site. Net benefit is true positives per patient minus false positives per patient weighted by the odds of the threshold probability, here 10%. For method 4 the violation rate is the proportion of all draws in which a threshold was certified and its observed PPV fell below v; for method 5 the false discovery proportion is the realised share of false positives among flagged patients. Discrimination (AUROC) and net benefit are reported per transfer in [`results/ppv_repairs.json`](results/ppv_repairs.json).

### 2.4 Post hoc analyses

Four measures were added after the protocol was fixed: the mean number of cases in each labelled sample, the proportion of draws in which the refitted threshold flagged every patient, the refitted threshold's error and sensitivity over the draws in which it was finite, and two net-benefit references per transfer, flagging every patient and the target's own threshold for 90% sensitivity fitted on all its labels. The second reference is an operating point, not the highest attainable net benefit. The proportion of draws flagging every patient was made a co-primary outcome. Intervals on the pooled median errors are the 2.5th and 97.5th centiles of 200 bootstrap redraws, resampling with replacement the transfers of a site and then the draws within each transfer; the interval on the label-free median also resamples patients, as for the falsification criteria. With 200 redraws, each bound is approximate to about one decimal place. The references are read on the whole target, the methods on the evaluation patients of each draw, and the refitted threshold's net benefit on the draws in which it was placed.

### 2.5 Falsification criteria

Three criteria were fixed with the protocol. The label-free correction would be judged sufficient outside Columbia if the upper bound of the bootstrap interval of its median error, obtained by resampling transfers and patients with replacement, remained below two points at both Beth Israel and Chongqing. Certification would be judged invalid under realistic sampling if its violation rate exceeded 15% at n = 100 at any site. The budget would be judged insufficient to certify v if a threshold was returned in fewer than half of draws.

## 3. Results

### 3.1 Without labels, the prevalence correction erred by more than ten points

The prevalence correction erred by a median of 13.1 points at Columbia, 11.4 points at Beth Israel and 51.7 points at Chongqing. It understated the observed PPV in every Columbia transfer and overstated it at Chongqing. At the source threshold, which this correction retains, sensitivity in the target was 82% at Columbia, 75% at Beth Israel and 84% at Chongqing, against 90% in the source. The label-free estimate for Columbia published in [1], over that study's selection of transfers, erred by 12.6 points. In the in-distribution controls the correction erred by 2.4 points, and with the true prevalence in place of the estimate Bayes' theorem erred by 0.5 points, against 0.2 points over the EchoNext controls of [1]. At the eight reserve sites it erred by 12.2 to 93.3 points.

### 3.2 With 100 labels, the refitted threshold mostly flagged every patient

With 25 labels, recalibration erred by 5.8, 8.5 and 9.3 points at Columbia, Beth Israel and Chongqing, and threshold refitting by 6.0, 5.7 and 6.6 points. With 100 labels the two errors were 3.1 and 2.6 points at Columbia, 5.7 and 2.2 points at Beth Israel, and 3.8 and 3.5 points at Chongqing; with 400 labels they were 1.9 and 1.6 points, 3.5 and 1.7 points, and 1.8 and 2.6 points. From 100 labels onwards the stated PPV exceeded the observed PPV in 44% to 55% of draws; with fewer labels recalibration overstated it in up to 72% of draws.

Recalibration retains the source threshold, and its sensitivity equalled that of the prevalence correction at every budget. Refitted thresholds flagged 99%, 99% and 98% of cases with 100 labels, and 94%, 95% and 92% with 400, against a target of 90%. Table 1 gives the refitted threshold by budget. With 100 labels it flagged every patient in 79%, 84% and 71% of the draws in which it was placed. Over the draws in which its threshold was finite, its error was 5.1, 4.3 and 5.5 points, above that of recalibration at Columbia and Chongqing, and its sensitivity was 92%, 94% and 93%. The share of draws flagging every patient fell below one half from 200 labels at Chongqing and from 400 labels at Columbia and Beth Israel. In the in-distribution controls the refitted threshold flagged every patient in 75% of the draws with 100 labels. At the median prevalence of the transfers, a labelled half is expected to hold nine cases from 272 labels at Columbia, 314 at Beth Israel and 121 at Chongqing.

Table 1. Refitted threshold by label budget at the three sites and in the in-distribution controls. Errors are median absolute errors of the stated PPV in percentage points. The error over finite thresholds pools the draws in which the threshold was finite, which gives more weight to transfers of higher prevalence; its 95% interval comes from the bootstrap of section 2.4. Sensitivity over finite thresholds is averaged over the transfers that had one, each transfer counting once. Mean case counts exceed those expected at the median prevalence because a few transfers of high prevalence raise them. With 25 labels the Columbia value rests on 18 draws and no Chongqing threshold was finite; with 50 labels the Chongqing value rests on 3 draws.

| Site | Labels | Mean cases | Draws with a refitted threshold | Of these, flagging every patient | Error, draws with a threshold | Error, finite threshold | Sensitivity, finite threshold |
|---|---|---|---|---|---|---|---|
| Columbia | 25 | 2.5 | 59% | 100% | 6.0 | 17.8 (9.8 to 21.0) | 92% |
| Columbia | 50 | 5.0 | 75% | 93% | 3.4 | 7.4 (6.7 to 8.5) | 91% |
| Columbia | 100 | 9.9 | 89% | 79% | 2.6 | 5.1 (4.8 to 5.4) | 92% |
| Columbia | 200 | 19.9 | 97% | 60% | 1.8 | 3.5 (3.3 to 3.7) | 91% |
| Columbia | 400 | 39.8 | 100% | 32% | 1.6 | 2.3 (2.1 to 2.5) | 91% |
| Beth Israel | 25 | 3.3 | 53% | 87% | 5.7 | 10.3 (8.7 to 11.5) | 92% |
| Beth Israel | 50 | 6.6 | 71% | 80% | 2.7 | 6.7 (5.8 to 7.7) | 94% |
| Beth Israel | 100 | 13.2 | 87% | 84% | 2.2 | 4.3 (3.7 to 5.1) | 94% |
| Beth Israel | 200 | 26.4 | 96% | 77% | 1.8 | 3.4 (2.9 to 4.6) | 92% |
| Beth Israel | 400 | 52.7 | 100% | 35% | 1.7 | 2.9 (2.3 to 3.6) | 92% |
| Chongqing | 25 | 3.8 | 89% | 100% | 6.6 | – | – |
| Chongqing | 50 | 7.6 | 98% | 98% | 5.9 | 11.0 (8.3 to 12.4) | 94% |
| Chongqing | 100 | 14.8 | 100% | 71% | 3.5 | 5.5 (4.2 to 6.9) | 93% |
| Chongqing | 200 | 30.0 | 100% | 3% | 3.0 | 3.2 (2.6 to 3.6) | 93% |
| Chongqing | 400 | 59.7 | 100% | 0% | 2.6 | 2.6 (2.2 to 2.9) | 92% |
| In-distribution controls | 25 | 3.1 | 59% | 96% | 6.3 | 6.5 (5.2 to 13.0) | 90% |
| In-distribution controls | 50 | 6.2 | 74% | 88% | 3.8 | 5.8 (4.5 to 7.2) | 91% |
| In-distribution controls | 100 | 12.4 | 87% | 75% | 2.4 | 5.0 (4.3 to 5.7) | 92% |
| In-distribution controls | 200 | 24.8 | 94% | 54% | 1.9 | 3.5 (3.2 to 3.8) | 91% |
| In-distribution controls | 400 | 49.6 | 98% | 32% | 1.6 | 2.5 (2.3 to 2.8) | 91% |

### 3.3 Certification met its marginal guarantee but rarely returned a threshold

At n = 100, the violation rate was 2.2% at Columbia, 1.4% at Beth Israel and 1.0% at Chongqing, all below δ, and a threshold was certified in 16.8%, 25.1% and 1.0% of draws. Among the draws in which a threshold was certified, its observed PPV fell below v in 13.2%, 5.7% and 100%; at Columbia this proportion exceeded δ at every budget, from 19.2% with 25 labels to 13.6% with 400. The Chongqing proportion comes from two draws. At most targets, no threshold in the grid attains the source PPV. Where a threshold was certified, sensitivity was 43% at Columbia, 87% at Beth Israel and 38% at Chongqing.

### 3.4 Weighted conformal selection exceeded its nominal level at Chongqing

The mean false discovery proportion was 0.75 at Chongqing against a nominal level of 0.39, 0.64 against 0.77 at Columbia and 0.46 against 0.60 at Beth Israel. Sensitivity was 64%, 68% and 86% at the three sites.

### 3.5 Reserve sites and net benefit

Table 2 gives the errors and Table 3 the net benefit at n = 100 for every site, the secondary reserve and the in-distribution controls included. With 100 labels the labelled samples held fewer than two cases on average at Shandong, CODE-15% and ZZU, and fewer than 16 at every other site. The refitted threshold flagged every patient in 71% to 100% of the draws in which it was placed, and in all of them at Shandong, CODE-15%, the CODE test set and ZZU. Its net benefit stayed within 0.006 of that of flagging every patient at every site except PTB-XL, where it was 0.009 lower. At Columbia its net benefit was 0.001, against −0.001 for flagging every patient and 0.044 for recalibration. The source threshold, which the prevalence correction and recalibration keep, had a higher net benefit than flagging every patient at every site.

Table 2. Median absolute error of the stated PPV, in percentage points, with 95% intervals, at n = 100 labels.

| Site | Transfers | Cases in 100 labels | Prevalence correction, no labels | Recalibration | Refitted threshold | Draws with a refitted threshold | Of these, flagging every patient |
|---|---|---|---|---|---|---|---|
| Columbia | 76 | 9.9 | 13.1 (10.7 to 17.1) | 3.1 (2.9 to 3.5) | 2.6 (2.3 to 2.8) | 89% | 79% |
| Beth Israel | 28 | 13.2 | 11.4 (6.7 to 30.3) | 5.7 (4.3 to 7.8) | 2.2 (1.8 to 2.3) | 87% | 84% |
| Chongqing | 1 | 14.8 | 51.7 (50.4 to 52.8) | 3.8 (3.3 to 4.3) | 3.5 (3.1 to 4.3) | 100% | 71% |
| Shandong | 25 | 1.6 | 22.3 | 4.9 (2.7 to 7.4) | 1.2 (1.0 to 1.3) | 48% | 100% |
| Chapman-Ningbo | 28 | 5.0 | 17.3 | 5.1 (3.8 to 7.0) | 1.9 (1.8 to 2.2) | 75% | 88% |
| CPSC | 28 | 10.3 | 12.2 | 5.1 (3.9 to 6.6) | 2.9 (2.5 to 3.2) | 93% | 80% |
| Georgia | 28 | 6.9 | 20.0 | 5.6 (4.3 to 7.5) | 2.2 (2.1 to 2.5) | 91% | 93% |
| PTB-XL | 28 | 15.4 | 14.8 | 4.7 (3.1 to 6.8) | 2.6 (2.6 to 2.7) | 83% | 82% |
| CODE-15% | 24 | 1.7 | 93.3 | 3.3 (2.2 to 5.4) | 1.6 (1.5 to 1.6) | 58% | 100% |
| CODE test set | 21 | 3.7 | 54.0 | 5.1 (3.5 to 8.0) | 1.9 (1.7 to 2.0) | 85% | 100% |
| ZZU | 14 | 1.2 | 22.5 | 6.9 (2.5 to 13.0) | 1.3 (0.8 to 1.6) | 44% | 100% |
| In-distribution controls | 77 | 12.4 | 2.4 | 3.0 (2.5 to 3.4) | 2.4 (2.2 to 2.8) | 87% | 75% |

Table 3. Net benefit at a threshold probability of 10%, at n = 100 labels. Recalibration keeps the source threshold and shares the prevalence correction's column; the certified threshold is averaged over the draws in which one was found.

| Site | Flag every patient | Target's own threshold | Prevalence correction and recalibration | Refitted threshold | Certified threshold | Weighted conformal selection |
|---|---|---|---|---|---|---|
| Columbia | −0.001 | 0.032 | 0.044 | 0.001 | 0.033 | 0.035 |
| Beth Israel | 0.036 | 0.101 | 0.090 | 0.031 | 0.118 | 0.073 |
| Chongqing | 0.055 | 0.081 | 0.085 | 0.060 | 0.050 | 0.084 |
| Shandong | −0.093 | 0.003 | 0.006 | −0.093 | 0.009 | −0.014 |
| Chapman-Ningbo | −0.055 | 0.013 | 0.026 | −0.052 | 0.027 | −0.001 |
| CPSC | 0.004 | 0.057 | 0.055 | 0.009 | 0.067 | 0.048 |
| Georgia | −0.034 | 0.037 | 0.037 | −0.032 | 0.034 | 0.005 |
| PTB-XL | 0.061 | 0.124 | 0.088 | 0.052 | 0.120 | 0.047 |
| CODE-15% | −0.092 | −0.077 | −0.041 | −0.092 | −0.036 | −0.032 |
| CODE test set | −0.070 | −0.009 | 0.000 | −0.070 | −0.009 | −0.011 |
| ZZU | −0.098 | −0.015 | −0.001 | −0.098 | 0.004 | −0.012 |
| In-distribution controls | 0.027 | 0.070 | 0.069 | 0.027 | 0.051 | 0.049 |

### 3.6 Falsification criteria

The upper bounds of the label-free median error were 30.3 points at Beth Israel and 52.8 points at Chongqing; with a single transfer, the Chongqing bound reflects the resampling of patients only. The certification violation rate did not exceed 15% at any site. A threshold was certified in fewer than half of draws at all three sites.

## 4. Discussion

### 4.1 Principal findings

Without labels, the prevalence correction did not state the PPV at any of the three sites, and the falsification criterion indicates that the need for local labels is not specific to Columbia. With 100 labels, recalibration stated it within a few points while keeping the source threshold and its sensitivity. The refitted threshold reached a smaller error at the same budget, but in most draws it flagged every patient, so that the stated and observed PPV both approached the prevalence. Over the draws in which its threshold was finite, its error was larger than that of recalibration at two of the three sites. The in-distribution controls showed a similar proportion of draws flagging every patient, which indicates that it follows from the number of cases and not from the shift between sites. Certification held its marginal guarantee, but 100 labels did not certify the source PPV at these targets, and at Columbia the certified thresholds that were found fell below v more often than δ.

### 4.2 Comparison with prior work

Without labels, the error of the prevalence correction at Columbia, 13.1 points, was close to the 12.6 points of the label-free estimate published in [1] over that study's selection of transfers. In [1], a threshold recalibrated on 858 separate outpatients at Columbia reached a sensitivity of 86.3%; the present study measures how many labels each updating method needs to state the PPV.

### 4.3 Strengths and limitations

The target v was fixed from the source scores alone, before any method was run, and the in-distribution controls separate the effect of the number of cases from that of the shift between sites. The comparison rests on three sites with correlated transfers within each, and the Beth Israel labels are machine statements. Chongqing contributes a single transfer. The observed PPV of each draw is estimated on a finite evaluation set, which adds sampling variance to the measured error and to the violation rate. The binomial tests of certification treat ECGs as independent although several may come from one patient, so its guarantee holds only approximately. Method 5 uses the model's score as its only covariate and therefore assumes that the diagnosis depends on the ECG only through that score; the false discovery proportion at Chongqing suggests that this assumption does not hold there. The target v is the source PPV, and a site that sets another target would obtain other certification rates and other sensitivities. Calibration curves are not shown; the calibration of the recalibrated model at 100 labels is in [`results/repairs.json`](results/repairs.json).

### 4.4 Implications

A site with about 100 labels can state its PPV by recalibration at the source threshold. A refitted threshold needs nine cases in the half used to fit it, which at the median prevalence of these transfers means 121 to 314 labels, and its error is informative only when the proportion of draws flagging every patient is reported with it.

## Declarations

The declarations of [1] apply to this study.

## References

1. Abbou R. Predicting an ECG model's positive predictive value from a clinic's prevalence: a retrospective measurement across hospitals and care settings. September 2026. [REPORT.md](REPORT.md), with its supplement, [SUPPLEMENT.md](SUPPLEMENT.md).
2. Saerens M, Latinne P, Decaestecker C. Adjusting the outputs of a classifier to new a priori probabilities: a simple procedure. *Neural Comput* 2002;14(1):21-41. doi:10.1162/089976602753284446.
3. Angelopoulos AN, Bates S, Candès EJ, Jordan MI, Lei L. Learn then Test: calibrating predictive algorithms to achieve risk control. arXiv:2110.01052, 2021.
4. Jin Y, Candès EJ. Model-free selective inference under covariate shift via weighted conformal p-values. arXiv:2307.09291, 2023.

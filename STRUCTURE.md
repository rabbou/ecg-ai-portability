# Label-free covariate-shift weighting of conformal thresholds for structural heart disease across care settings at Columbia

At Columbia, a threshold set to detect 90% of structural heart disease (SHD) among inpatients detects 71.6% of outpatient cases for the study's trained ResNet ([README](README.md)). Weighted conformal prediction (Tibshirani et al., arXiv:1904.06019) can restore coverage without target labels when the probability of disease given the tracing, P(Y|X), is the same in both populations. Whether it is the same between care settings of one hospital has not been measured for any ECG model of structural heart disease. We reweighted the per-label conformal threshold of four models on EchoNext with weights estimated from unlabelled outpatients, on three covariate sets, and judged the repair against a decision rule fixed before any result was read. Weighting on a 1,024-dimensional ECG embedding brings composite sensitivity among outpatients from 71.6% to between 87.8% and 89.7% across the four models, against 72.7% to 81.5% when weighting on age and sex. The pre-specified criterion is not met by any of the four: the lower Wilson bound at the effective sample size stays below 85%, and only the frozen ECGFounder passes a strict majority of judged anomalies. The three fine-tuned ECGFounder models of arm A1 enter these tables when their training ends.

## Methods

### Participants and outcome

EchoNext v1.1.1 (PhysioNet, restricted licence) holds 100,000 Columbia ECGs, each paired with a transthoracic echocardiogram within a year, eleven echocardiographic findings and their composite (SHD). The published splits are used unchanged: 72,475 training ECGs; 1,903 inpatient ECGs of the validation split for calibration; and, in the test split, 2,203 inpatients, 1,971 emergency patients and 1,059 outpatients, one ECG per patient, no patient in two roles (`ecs.echonext.transfer_cohorts` raises otherwise). The outpatient test cohort holds 271 composite cases. This is a retrospective study reported following TRIPOD+AI (Collins et al., BMJ 2024); no ECG, score or weight derived from EchoNext leaves the analysis machine (licence clause 3).

### Models

Four models whose scores the published study already holds: the study's ResNet trained on EchoNext; the same ResNet frozen at a seeded random initialisation with one logistic probe per label; ECGFounder (Net1D, 30.8 million parameters) frozen with the same probes; and the published EchoNext mini-model on its own weights. Arm A1 fine-tunes ECGFounder end to end (`scripts/echonext_finetune.py`): the published weights, a twelve-output head, AdamW at a learning rate of 1e-4 with weight decay 0.01 on a cosine schedule, batch 64, ten epochs, three seeds, the epoch chosen on the tenth of training patients the ResNet held out. It runs on the Apple M5 GPU at 46.5 ECGs per second under the repository's torch 2.2.2.

### Weights without target labels

For each target care setting, a domain classifier separates the calibration inpatients from the unlabelled target ECGs: an L2 logistic regression on standardised covariates, its penalty chosen by inner five-fold cross-validation on log loss, its predictions cross-fitted over five folds. The weight w(x) is the classifier's odds rescaled by the cohort sizes, capped at the 99th centile of the calibration weights. Three covariate sets are compared: age and sex, the low reference, because P(Y|X) is not plausibly invariant given them alone; the seven ECG measurements of the distribution (sex, ventricular and atrial rates, PR, QRS and QTc intervals, age); and the frozen ECGFounder embedding. Under invariant P(Y|X), the density ratio between populations at a fixed diagnosis equals w(x) up to a constant, so each class's conformal quantile is weighted by w(x) alone and the constant cancels. The weights are shared by all models, so models differ by their scores only.

Domain classifier AUROC, cross-fitted, by target and covariate set:

<!-- table:domain -->
| Target | age, sex | ECG measurements | embedding |
|---|---|---|---|
| inpatient | 0.499 | 0.504 | 0.485 |
| emergency | 0.552 | 0.590 | 0.768 |
| outpatient | 0.576 | 0.728 | 0.836 |
<!-- /table -->

The inpatient row compares two draws of one population and is a negative control.

### Pre-specified decision rules

Fixed in the protocol of 16 October 2026 before any result was read. An anomaly is judged when it counts at least 50 cases among test outpatients; four do (LVEF at most 45%, 70 cases; LV wall thickness at least 13 mm, 146; moderate or greater aortic stenosis, 51; PASP at least 45 mmHg, 63). The repair holds when the composite's 95% Wilson interval lies within [85%, 95%] and a strict majority of judged anomalies clears 80% in a one-sided score test at 0.025 with Holm's correction, on the embedding covariate. Both are computed at the effective size, the smaller of the target case count and the Kish effective size of the calibration cases' weights, because the weighted threshold rests on those calibration cases. The fine-tuned arm halves the published loss when the upper bound of its composite sensitivity loss among outpatients, 90% minus the lower Wilson bound, is at most 9 points, for each of the three seeds.

### Calibration and selective prediction

For each model, finding and care setting: calibration slope and calibration-in-the-large, the integrated calibration index (ICI; Austin and Steyerberg, Stat Med 2019, here with a Gaussian-kernel smoother on logit p in place of loess), and the Brier score. The decision is the flag at the calibration threshold that detects 90% of inpatient cases; confidence is the distance from that threshold on the logit scale. Abstaining on the least confident ECGs traces the risk-coverage curve, summarised by its generalised area (AUGRC; Traub et al., arXiv:2407.01032) and by the sensitivity kept when 80% of ECGs are retained. Intervals are 2.5th and 97.5th percentiles of 500 bootstrap draws of target ECGs, the threshold held fixed.

## Results

### Composite sensitivity among outpatients

Sensitivity with its 95% Wilson interval; for the weighted rules, the interval and n are at the effective size.

<!-- table:composite -->
| Arm | Single threshold | Weighted on age, sex | Weighted on ECG measurements | Weighted on embedding |
|---|---|---|---|---|
| ResNet trained on EchoNext | 71.6% (65.9%–76.6%) | 74.2% (68.6%–79.0%, n=271) | 79.3% (74.1%–83.7%, n=271) | 89.3% (84.3%–92.7%, n=213) |
| ResNet at random initialisation, probes | 78.6% (73.3%–83.1%) | 81.5% (76.5%–85.7%, n=271) | 83.8% (78.9%–87.7%, n=271) | 87.8% (82.7%–91.5%, n=213) |
| ECGFounder frozen, probes | 71.6% (65.9%–76.6%) | 72.7% (67.1%–77.7%, n=271) | 78.6% (73.3%–83.1%, n=271) | 89.7% (84.9%–93.1%, n=213) |
| EchoNext mini-model | 72.7% (67.1%–77.7%) | 73.8% (68.3%–78.7%, n=271) | 82.3% (77.3%–86.4%, n=271) | 89.7% (84.9%–93.1%, n=213) |
<!-- /table -->

Weighting on age and sex moves sensitivity by 3 points at most, and weighting on the ECG measurements recovers part of it. Weighting on the embedding brings every model within a few points of 90%, with an effective size of 213 calibration cases out of the inpatient cohort's.

### Judged anomalies, weighted on the embedding

Sensitivity among outpatients, effective size, and the Holm-corrected test against 80%.

<!-- table:anomalies -->
| Arm | LVEF ≤ 45% | LV wall ≥ 13 mm | Aortic stenosis | PASP ≥ 45 mmHg | Repair |
|---|---|---|---|---|---|
| ResNet trained on EchoNext | 88.6%, n=70, fails | 89.7%, n=101, passes | 100.0%, n=29, passes | 82.5%, n=63, fails | fails |
| ResNet at random initialisation, probes | 87.1%, n=70, fails | 91.1%, n=101, passes | 94.1%, n=29, fails | 95.2%, n=63, passes | fails |
| ECGFounder frozen, probes | 92.9%, n=70, passes | 95.9%, n=101, passes | 100.0%, n=29, passes | 90.5%, n=63, passes | fails |
| EchoNext mini-model | 84.3%, n=70, fails | 91.8%, n=101, passes | 98.0%, n=29, passes | 84.1%, n=63, fails | fails |
<!-- /table -->

### Per-label thresholds refitted on local labels

Composite sensitivity on a held-out half of the outpatients when the per-label thresholds are refitted on n labelled outpatients drawn from the other half: mean and 10th to 90th percentile of 200 draws (the mean can fall outside that range when a tenth of the draws flag almost everyone). With 25 or 50 labels, few cases are drawn and the threshold often flags every ECG.

<!-- table:ladder -->
| Arm | 25 labels | 50 labels | 100 labels | 200 labels | 400 labels |
|---|---|---|---|---|---|
| ResNet trained on EchoNext | 98.9% (99.3%–100.0%) | 94.5% (89.1%–100.0%) | 93.4% (89.7%–99.3%) | 92.7% (90.5%–95.6%) | 92.1% (92.0%–92.0%) |
| ResNet at random initialisation, probes | 98.5% (94.2%–100.0%) | 90.7% (81.8%–100.0%) | 88.8% (83.9%–94.2%) | 87.3% (84.7%–89.1%) | 86.7% (85.4%–87.6%) |
| ECGFounder frozen, probes | 99.3% (100.0%–100.0%) | 96.0% (88.2%–100.0%) | 95.3% (88.3%–100.0%) | 95.2% (91.2%–99.3%) | 94.8% (92.7%–95.6%) |
| EchoNext mini-model | 99.2% (99.3%–100.0%) | 96.8% (93.5%–100.0%) | 97.6% (94.9%–99.3%) | 97.3% (94.9%–99.3%) | 97.3% (94.9%–99.3%) |
<!-- /table -->

### Calibration and selective prediction on the composite

<!-- table:calibration -->
| Arm | Target | Slope | Intercept | ICI | Brier | AUGRC | Sens. at 80% kept |
|---|---|---|---|---|---|---|---|
| ResNet trained on EchoNext | inpatient | 0.74 (0.68–0.81) | -0.25 (-0.36–-0.15) | 0.076 (0.061–0.092) | 0.180 (0.170–0.190) | 0.123 (0.114–0.133) | 95.1% (93.7%–96.2%) |
| ResNet trained on EchoNext | outpatient | 0.82 (0.71–0.94) | -0.41 (-0.59–-0.23) | 0.069 (0.052–0.090) | 0.138 (0.125–0.153) | 0.108 (0.094–0.120) | 76.3% (71.6%–82.2%) |
| ResNet at random initialisation, probes | inpatient | 0.98 (0.88–1.10) | -0.11 (-0.20–-0.02) | 0.023 (0.014–0.043) | 0.192 (0.184–0.200) | 0.138 (0.128–0.149) | 94.8% (93.4%–96.0%) |
| ResNet at random initialisation, probes | outpatient | 1.08 (0.89–1.27) | -0.74 (-0.89–-0.62) | 0.139 (0.120–0.162) | 0.173 (0.164–0.185) | 0.168 (0.152–0.184) | 81.3% (76.4%–86.0%) |
| ECGFounder frozen, probes | inpatient | 0.88 (0.80–0.95) | -0.16 (-0.26–-0.06) | 0.045 (0.032–0.062) | 0.182 (0.174–0.191) | 0.124 (0.115–0.135) | 93.8% (92.1%–95.2%) |
| ECGFounder frozen, probes | outpatient | 0.90 (0.77–1.04) | -0.43 (-0.59–-0.28) | 0.069 (0.053–0.093) | 0.148 (0.136–0.161) | 0.112 (0.098–0.125) | 76.8% (71.4%–82.5%) |
| EchoNext mini-model | inpatient | 0.98 (0.89–1.07) | 0.18 (0.08–0.26) | 0.032 (0.022–0.049) | 0.184 (0.176–0.194) | 0.131 (0.122–0.142) | 95.5% (93.9%–96.6%) |
| EchoNext mini-model | outpatient | 1.04 (0.91–1.20) | -0.16 (-0.33–-0.02) | 0.035 (0.023–0.058) | 0.141 (0.129–0.155) | 0.118 (0.104–0.133) | 78.4% (72.6%–83.3%) |
<!-- /table -->

Retaining the 80% most confident ECGs keeps more than 93% sensitivity among inpatients for every model, and between 76.3% and 81.3% among outpatients: the abstention rule set on inpatients does not set aside the outpatient cases the threshold misses.

### Arm A1, the loss of composite sensitivity among outpatients

<!-- table:a1 -->
| Arm | Outpatient sensitivity | Loss, points | Loss at most 9 points |
|---|---|---|---|
| ResNet trained on EchoNext | 71.6% (low 65.9%) | 18.4 (up to 24.1) | no |
| ResNet at random initialisation, probes | 78.6% (low 73.3%) | 11.4 (up to 16.7) | no |
| ECGFounder frozen, probes | 71.6% (low 65.9%) | 18.4 (up to 24.1) | no |
| EchoNext mini-model | 72.7% (low 67.1%) | 17.3 (up to 22.9) | no |
<!-- /table -->

## Discussion

Falsifier A2 is not met for the four models measured: weighting on the ECG embedding restores composite sensitivity among outpatients to 87.8% to 89.7%, but no model's lower Wilson bound at 213 effective cases reaches 85%, and only the frozen ECGFounder clears 80% on a majority of the judged anomalies. The result is compatible with P(Y|X) changing between care settings, or with an embedding that does not hold the covariates on which it changes; the bound of Han et al. (arXiv:2607.18088) on label-free class-conditional coverage is consistent with it and is not demonstrated by it. The size of the recovery from age and sex to the embedding suggests that most of the loss follows from a shift in ECG appearance rather than in demographics. A repair from 100 to 200 local labels reaches the 90% target for three of the four models, at the cost of labelling outpatients.

## Limits

The audit of Han et al. on target labels, named in the protocol as a third reference, is not implemented; the refitted thresholds on local labels stand in as the labelled reference. The decision rule's effective size is our reading of the protocol's "calculé sur la taille effective". The weights use the frozen ECGFounder embedding for every model, including models that do not read it. The Columbia to Boston arm (A4) awaits credentialed access to MIMIC-IV-Echo; Otabor et al. (Eur Heart J Digit Health 2026, doi:10.1093/ehjdh/ztag147) already report the EchoNext mini-model's discrimination and recalibration in 45,878 MIMIC-IV patients, without conformal thresholds or care settings. TRIPOD+AI items not covered here: sample size justification, handling of missing data (none in the covariates used), and a model-updating protocol for deployment.

## Literature search

Searched on 20 October 2026. arXiv API: "EchoNext AND conformal", 0 records; "EchoNext AND calibration", 1 (arXiv:2607.24553); "EchoNext", 6. medRxiv's search refused the request (HTTP 403); Europe PMC, which indexes medRxiv preprints: "EchoNext AND conformal", 0 records; "EchoNext AND calibration", 14, of which Otabor et al. above is the only external validation with recalibration. Colebank (arXiv:2605.22968) studies Bayesian uncertainty for triage on EchoNext, without conformal thresholds or care settings in its abstract.

## Reproduce

```bash
# On the machine holding EchoNext and the four arms' scores (~/data/echonext-derived/scores)
PYTORCH_ENABLE_MPS_FALLBACK=1 /usr/bin/lockf ~/.claude/locks/mac-gpu.lock \
  .venv/bin/python scripts/echonext_finetune.py --seeds 0 1 2
.venv/bin/python scripts/structure.py           # results/structure*.{csv,json}, then the tables here
.venv/bin/python scripts/structure.py --pages   # the tables alone, from the committed results
```

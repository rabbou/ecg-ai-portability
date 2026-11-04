# Supplement: label-free recalibration of an ECG threshold for structural heart disease across care settings

The per-model results behind section 3 of the report. Each table is rendered from the committed result files; the sections of the report that cite them are named in each caption.

## S1. Composite sensitivity among outpatients

Table S1. Composite sensitivity among outpatients with its 95% Wilson interval (section 3.2). For the weighted thresholds, the interval and n are at the effective size under the primary reading.

<!-- table:composite -->
| Arm | Inpatient threshold | Weighted on age, sex | Weighted on ECG measurements | Weighted on embedding |
|---|---|---|---|---|
| ResNet trained on EchoNext | 71.6% (65.9%–76.6%) | 74.2% (68.6%–79.0%, n=271) | 79.3% (74.1%–83.7%, n=271) | 89.3% (84.3%–92.7%, n=213) |
| ResNet at random initialisation, probes | 78.6% (73.3%–83.1%) | 81.5% (76.5%–85.7%, n=271) | 83.8% (78.9%–87.7%, n=271) | 87.8% (82.7%–91.5%, n=213) |
| ECGFounder frozen, probes | 71.6% (65.9%–76.6%) | 72.7% (67.1%–77.7%, n=271) | 78.6% (73.3%–83.1%, n=271) | 89.7% (84.9%–93.1%, n=213) |
| EchoNext mini-model | 72.7% (67.1%–77.7%) | 73.8% (68.3%–78.7%, n=271) | 82.3% (77.3%–86.4%, n=271) | 89.7% (84.9%–93.1%, n=213) |
| ECGFounder fine-tuned, seed 0 | 72.7% (67.1%–77.7%) | 73.8% (68.3%–78.7%, n=271) | 80.1% (74.9%–84.4%, n=271) | 87.1% (81.7%–90.7%, n=213) |
| ECGFounder fine-tuned, seed 1 | 72.0% (66.3%–77.0%) | 73.4% (67.9%–78.3%, n=271) | 81.5% (76.5%–85.7%, n=271) | 88.2% (83.2%–91.9%, n=213) |
| ECGFounder fine-tuned, seed 2 | 75.6% (70.2%–80.4%) | 77.9% (72.5%–82.4%, n=271) | 81.2% (76.1%–85.4%, n=271) | 89.3% (84.3%–92.7%, n=213) |
<!-- /table -->

## S2. Judged findings among outpatients

Table S2. Sensitivity among outpatients for each judged finding, weighted on the embedding, with the effective size under the primary reading (section 3.2). Findings with an asterisk are those whose Holm-corrected test rejects a sensitivity of 80%. Criterion: both conditions of section 2.5.

<!-- table:anomalies -->
| Arm | LVEF ≤ 45% | LV wall ≥ 13 mm | Aortic stenosis | PASP ≥ 45 mmHg | Criterion |
|---|---|---|---|---|---|
| ResNet trained on EchoNext | 88.6% (n=70) | 89.7%* (n=101) | 100.0%* (n=29) | 82.5% (n=63) | not met |
| ResNet at random initialisation, probes | 87.1% (n=70) | 91.1%* (n=101) | 94.1% (n=29) | 95.2%* (n=63) | not met |
| ECGFounder frozen, probes | 92.9%* (n=70) | 95.9%* (n=101) | 100.0%* (n=29) | 90.5%* (n=63) | not met |
| EchoNext mini-model | 84.3% (n=70) | 91.8%* (n=101) | 98.0%* (n=29) | 84.1% (n=63) | not met |
| ECGFounder fine-tuned, seed 0 | 94.3%* (n=70) | 91.8%* (n=101) | 100.0%* (n=29) | 81.0% (n=63) | not met |
| ECGFounder fine-tuned, seed 1 | 91.4% (n=70) | 89.0% (n=101) | 100.0%* (n=29) | 82.5% (n=63) | not met |
| ECGFounder fine-tuned, seed 2 | 92.9%* (n=70) | 89.7%* (n=101) | 100.0%* (n=29) | 85.7% (n=63) | not met |
<!-- /table -->

## S3. Composite sensitivity among emergency patients

Table S3. Composite sensitivity among emergency patients, as in Table S1.

<!-- table:composite_emergency -->
| Arm | Inpatient threshold | Weighted on age, sex | Weighted on ECG measurements | Weighted on embedding |
|---|---|---|---|---|
| ResNet trained on EchoNext | 86.2% (83.6%–88.4%) | 87.1% (84.5%–89.2%, n=788) | 87.9% (85.5%–90.0%, n=788) | 91.5% (88.8%–93.4%, n=571) |
| ResNet at random initialisation, probes | 84.4% (81.7%–86.8%) | 85.8% (83.2%–88.1%, n=788) | 86.7% (84.1%–88.9%, n=788) | 89.6% (86.9%–91.9%, n=571) |
| ECGFounder frozen, probes | 82.9% (80.1%–85.3%) | 83.6% (80.9%–86.0%, n=788) | 85.0% (82.4%–87.3%, n=788) | 88.5% (85.6%–90.8%, n=571) |
| EchoNext mini-model | 85.5% (82.9%–87.8%) | 86.2% (83.6%–88.4%, n=788) | 87.2% (84.7%–89.3%, n=788) | 89.6% (86.9%–91.9%, n=571) |
| ECGFounder fine-tuned, seed 0 | 82.6% (79.8%–85.1%) | 83.1% (80.3%–85.6%, n=788) | 84.8% (82.1%–87.1%, n=788) | 88.7% (85.9%–91.1%, n=571) |
| ECGFounder fine-tuned, seed 1 | 85.3% (82.6%–87.6%) | 85.7% (83.0%–87.9%, n=788) | 87.6% (85.1%–89.7%, n=788) | 90.7% (88.1%–92.8%, n=571) |
| ECGFounder fine-tuned, seed 2 | 84.9% (82.2%–87.2%) | 85.4% (82.8%–87.7%, n=788) | 87.6% (85.1%–89.7%, n=788) | 89.1% (86.3%–91.4%, n=571) |
<!-- /table -->

## S4. Judged findings among emergency patients

Table S4. Judged findings among emergency patients, as in Table S2.

<!-- table:anomalies_emergency -->
| Arm | LVEF ≤ 45% | LV wall ≥ 13 mm | Aortic stenosis | Mitral regurgitation | Tricuspid regurgitation | RV dysfunction | PASP ≥ 45 mmHg | TR velocity ≥ 3.2 m/s | Criterion |
|---|---|---|---|---|---|---|---|---|---|
| ResNet trained on EchoNext | 91.3%* (n=273) | 93.4%* (n=235) | 87.9% (n=66) | 92.4%* (n=85) | 93.1%* (n=99) | 94.5%* (n=129) | 95.3%* (n=174) | 85.0% (n=89) | met |
| ResNet at random initialisation, probes | 90.0%* (n=273) | 93.4%* (n=235) | 87.9% (n=66) | 89.1% (n=85) | 94.7%* (n=99) | 91.0%* (n=129) | 95.3%* (n=174) | 75.5% (n=89) | met |
| ECGFounder frozen, probes | 87.4%* (n=273) | 91.3%* (n=235) | 84.8% (n=66) | 83.2% (n=85) | 92.4%* (n=99) | 91.0%* (n=129) | 92.4%* (n=174) | 87.8% (n=89) | met |
| EchoNext mini-model | 86.8%* (n=273) | 91.3%* (n=235) | 84.8% (n=66) | 84.9% (n=85) | 95.4%* (n=99) | 89.7%* (n=129) | 94.9%* (n=174) | 87.1% (n=89) | met |
| ECGFounder fine-tuned, seed 0 | 88.7%* (n=273) | 95.4%* (n=235) | 92.4%* (n=66) | 90.8%* (n=85) | 91.6%* (n=99) | 91.0%* (n=129) | 91.6%* (n=174) | 83.7% (n=89) | met |
| ECGFounder fine-tuned, seed 1 | 89.7%* (n=273) | 95.7%* (n=235) | 86.4% (n=66) | 89.9%* (n=85) | 90.1%* (n=99) | 94.5%* (n=129) | 92.4%* (n=174) | 90.5%* (n=89) | met |
| ECGFounder fine-tuned, seed 2 | 88.1%* (n=273) | 91.3%* (n=235) | 87.9% (n=66) | 89.1% (n=85) | 90.1%* (n=99) | 93.1%* (n=129) | 92.0%* (n=174) | 83.0% (n=89) | met |
<!-- /table -->

## S5. The criterion under each reading of the effective size

Table S5. Bounds of the 95% Wilson interval of composite sensitivity weighted on the embedding, the size at which both conditions are judged, the number of judged findings whose test rejects 80%, and the criterion, under each reading (section 3.3).

<!-- table:readings -->
| Arm | Outpatient, smaller of cases and Kish size | Outpatient, target cases | Emergency, smaller of cases and Kish size | Emergency, target cases |
|---|---|---|---|---|
| ResNet trained on EchoNext | 84.3%–92.7% (n=213), 2 of 4 findings, not met | 85.1%–92.4% (n=271), 2 of 4 findings, not met | 88.8%–93.4% (n=571), 6 of 8 findings, met | 89.3%–93.2% (n=788), 6 of 8 findings, met |
| ResNet at random initialisation, probes | 82.7%–91.5% (n=213), 2 of 4 findings, not met | 83.4%–91.2% (n=271), 3 of 4 findings, not met | 86.9%–91.9% (n=571), 5 of 8 findings, met | 87.3%–91.5% (n=788), 6 of 8 findings, met |
| ECGFounder frozen, probes | 84.9%–93.1% (n=213), 4 of 4 findings, not met | 85.5%–92.8% (n=271), 4 of 4 findings, met | 85.6%–90.8% (n=571), 5 of 8 findings, met | 86.0%–90.5% (n=788), 5 of 8 findings, met |
| EchoNext mini-model | 84.9%–93.1% (n=213), 2 of 4 findings, not met | 85.5%–92.8% (n=271), 2 of 4 findings, not met | 86.9%–91.9% (n=571), 5 of 8 findings, met | 87.3%–91.5% (n=788), 5 of 8 findings, met |
| ECGFounder fine-tuned, seed 0 | 81.7%–90.7% (n=213), 3 of 4 findings, not met | 82.6%–90.6% (n=271), 3 of 4 findings, not met | 85.9%–91.1% (n=571), 7 of 8 findings, met | 86.3%–90.7% (n=788), 7 of 8 findings, met |
| ECGFounder fine-tuned, seed 1 | 83.2%–91.9% (n=213), 1 of 4 findings, not met | 83.8%–91.5% (n=271), 3 of 4 findings, not met | 88.1%–92.8% (n=571), 7 of 8 findings, met | 88.5%–92.6% (n=788), 7 of 8 findings, met |
| ECGFounder fine-tuned, seed 2 | 84.3%–92.7% (n=213), 3 of 4 findings, not met | 85.1%–92.4% (n=271), 3 of 4 findings, met | 86.3%–91.4% (n=571), 5 of 8 findings, met | 86.7%–91.1% (n=788), 6 of 8 findings, met |
<!-- /table -->

## S6. The composite flag per model

Table S6. Sensitivity, specificity, proportion of ECGs flagged and PPV of the composite flag at the inpatient threshold and at the threshold weighted on the embedding (section 3.4). Intervals for sensitivity are in Tables S1 and S3.

<!-- table:flags -->
| Arm | Target | Threshold | Sensitivity | Specificity | Flagged | PPV |
|---|---|---|---|---|---|---|
| ResNet trained on EchoNext | outpatient | inpatient | 71.6% | 71.1% (67.8%–74.1%) | 39.8% | 46.0% (41.3%–50.7%) |
| ResNet trained on EchoNext | outpatient | weighted on embedding | 89.3% | 42.5% (39.1%–46.0%) | 65.6% | 34.8% (31.4%–38.4%) |
| ResNet at random initialisation, probes | outpatient | inpatient | 78.6% | 57.4% (53.9%–60.8%) | 51.8% | 38.8% (34.8%–42.9%) |
| ResNet at random initialisation, probes | outpatient | weighted on embedding | 87.8% | 29.1% (26.0%–32.3%) | 75.3% | 29.9% (26.8%–33.1%) |
| ECGFounder frozen, probes | outpatient | inpatient | 71.6% | 71.3% (68.1%–74.4%) | 39.7% | 46.2% (41.5%–51.0%) |
| ECGFounder frozen, probes | outpatient | weighted on embedding | 89.7% | 42.5% (39.1%–46.0%) | 65.7% | 34.9% (31.5%–38.5%) |
| EchoNext mini-model | outpatient | inpatient | 72.7% | 71.1% (67.8%–74.1%) | 40.1% | 46.4% (41.7%–51.1%) |
| EchoNext mini-model | outpatient | weighted on embedding | 89.7% | 37.4% (34.1%–40.9%) | 69.5% | 33.0% (29.7%–36.5%) |
| ECGFounder fine-tuned, seed 0 | outpatient | inpatient | 72.7% | 76.0% (72.9%–78.9%) | 36.4% | 51.0% (46.1%–56.0%) |
| ECGFounder fine-tuned, seed 0 | outpatient | weighted on embedding | 87.1% | 49.1% (45.6%–52.6%) | 60.2% | 37.0% (33.4%–40.9%) |
| ECGFounder fine-tuned, seed 1 | outpatient | inpatient | 72.0% | 75.8% (72.6%–78.6%) | 36.4% | 50.5% (45.5%–55.5%) |
| ECGFounder fine-tuned, seed 1 | outpatient | weighted on embedding | 88.2% | 49.7% (46.3%–53.2%) | 60.0% | 37.6% (34.0%–41.5%) |
| ECGFounder fine-tuned, seed 2 | outpatient | inpatient | 75.6% | 72.6% (69.4%–75.6%) | 39.8% | 48.7% (44.0%–53.5%) |
| ECGFounder fine-tuned, seed 2 | outpatient | weighted on embedding | 89.3% | 45.8% (42.4%–49.3%) | 63.2% | 36.2% (32.6%–39.9%) |
| ResNet trained on EchoNext | emergency | inpatient | 86.2% | 61.0% (58.2%–63.8%) | 57.8% | 59.6% (56.7%–62.4%) |
| ResNet trained on EchoNext | emergency | weighted on embedding | 91.5% | 49.0% (46.2%–51.9%) | 67.2% | 54.5% (51.8%–57.1%) |
| ResNet at random initialisation, probes | emergency | inpatient | 84.4% | 54.5% (51.7%–57.3%) | 61.0% | 55.3% (52.5%–58.1%) |
| ResNet at random initialisation, probes | emergency | weighted on embedding | 89.6% | 41.0% (38.2%–43.8%) | 71.2% | 50.3% (47.7%–52.9%) |
| ECGFounder frozen, probes | emergency | inpatient | 82.9% | 60.0% (57.2%–62.8%) | 57.1% | 58.0% (55.1%–60.8%) |
| ECGFounder frozen, probes | emergency | weighted on embedding | 88.5% | 49.6% (46.8%–52.5%) | 65.6% | 53.9% (51.2%–56.6%) |
| EchoNext mini-model | emergency | inpatient | 85.5% | 57.7% (54.8%–60.4%) | 59.6% | 57.4% (54.5%–60.2%) |
| EchoNext mini-model | emergency | weighted on embedding | 89.6% | 49.5% (46.6%–52.3%) | 66.2% | 54.1% (51.4%–56.8%) |
| ECGFounder fine-tuned, seed 0 | emergency | inpatient | 82.6% | 65.4% (62.7%–68.1%) | 53.8% | 61.4% (58.4%–64.3%) |
| ECGFounder fine-tuned, seed 0 | emergency | weighted on embedding | 88.7% | 52.9% (50.1%–55.7%) | 63.7% | 55.7% (52.9%–58.4%) |
| ECGFounder fine-tuned, seed 1 | emergency | inpatient | 85.3% | 61.3% (58.5%–64.0%) | 57.3% | 59.5% (56.6%–62.3%) |
| ECGFounder fine-tuned, seed 1 | emergency | weighted on embedding | 90.7% | 48.7% (45.9%–51.5%) | 67.1% | 54.1% (51.4%–56.8%) |
| ECGFounder fine-tuned, seed 2 | emergency | inpatient | 84.9% | 61.9% (59.1%–64.6%) | 56.8% | 59.7% (56.8%–62.6%) |
| ECGFounder fine-tuned, seed 2 | emergency | weighted on embedding | 89.1% | 51.1% (48.3%–54.0%) | 64.9% | 54.8% (52.1%–57.6%) |
<!-- /table -->

## S7. Threshold recalibration on local labels

Table S7. Composite sensitivity in one half of the outpatients when the per-label thresholds were recalibrated on n labelled outpatients from the other half: mean and 10th to 90th percentiles over 200 draws (section 3.5). The mean can lie outside that range when a tenth of the draws flag nearly every ECG; with 25 or 50 outpatients, few cases are drawn and the threshold often flags every ECG.

<!-- table:ladder -->
| Arm | 25 labels | 50 labels | 100 labels | 200 labels | 400 labels |
|---|---|---|---|---|---|
| ResNet trained on EchoNext | 98.9% (99.3%–100.0%) | 94.5% (89.1%–100.0%) | 93.4% (89.7%–99.3%) | 92.7% (90.5%–95.6%) | 92.1% (92.0%–92.0%) |
| ResNet at random initialisation, probes | 98.5% (94.2%–100.0%) | 90.7% (81.8%–100.0%) | 88.8% (83.9%–94.2%) | 87.3% (84.7%–89.1%) | 86.7% (85.4%–87.6%) |
| ECGFounder frozen, probes | 99.3% (100.0%–100.0%) | 96.0% (88.2%–100.0%) | 95.3% (88.3%–100.0%) | 95.2% (91.2%–99.3%) | 94.8% (92.7%–95.6%) |
| EchoNext mini-model | 99.2% (99.3%–100.0%) | 96.8% (93.5%–100.0%) | 97.6% (94.9%–99.3%) | 97.3% (94.9%–99.3%) | 97.3% (94.9%–99.3%) |
| ECGFounder fine-tuned, seed 0 | 98.8% (97.8%–100.0%) | 93.7% (87.6%–100.0%) | 91.5% (88.3%–97.8%) | 90.6% (88.3%–94.9%) | 89.7% (89.1%–89.8%) |
| ECGFounder fine-tuned, seed 1 | 98.9% (96.4%–100.0%) | 93.8% (83.2%–100.0%) | 92.8% (83.9%–99.3%) | 91.5% (83.9%–95.6%) | 91.1% (90.7%–92.0%) |
| ECGFounder fine-tuned, seed 2 | 98.6% (97.7%–100.0%) | 92.5% (83.9%–100.0%) | 89.4% (83.9%–97.8%) | 88.2% (83.9%–92.0%) | 87.2% (86.9%–87.6%) |
<!-- /table -->

## S8. Calibration and selective prediction

Table S8. Calibration and selective prediction for the composite among inpatients and outpatients, with bootstrap intervals (section 3.6).

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
| ECGFounder fine-tuned, seed 0 | inpatient | 0.94 (0.86–1.02) | -0.39 (-0.49–-0.30) | 0.068 (0.056–0.085) | 0.179 (0.172–0.188) | 0.117 (0.108–0.128) | 93.8% (92.2%–95.2%) |
| ECGFounder fine-tuned, seed 0 | outpatient | 1.00 (0.86–1.18) | -0.66 (-0.82–-0.51) | 0.106 (0.085–0.127) | 0.146 (0.135–0.159) | 0.096 (0.084–0.111) | 77.7% (71.9%–83.1%) |
| ECGFounder fine-tuned, seed 1 | inpatient | 0.87 (0.80–0.94) | -0.13 (-0.24–-0.03) | 0.054 (0.041–0.070) | 0.175 (0.166–0.184) | 0.119 (0.111–0.130) | 93.9% (92.4%–95.4%) |
| ECGFounder fine-tuned, seed 1 | outpatient | 0.93 (0.80–1.07) | -0.31 (-0.47–-0.15) | 0.053 (0.036–0.075) | 0.136 (0.124–0.150) | 0.099 (0.086–0.113) | 78.1% (73.2%–83.7%) |
| ECGFounder fine-tuned, seed 2 | inpatient | 0.83 (0.76–0.90) | -0.05 (-0.15–0.05) | 0.050 (0.037–0.066) | 0.176 (0.168–0.186) | 0.122 (0.113–0.132) | 93.2% (91.6%–94.7%) |
| ECGFounder fine-tuned, seed 2 | outpatient | 0.97 (0.83–1.12) | -0.39 (-0.56–-0.23) | 0.063 (0.045–0.084) | 0.136 (0.124–0.149) | 0.114 (0.101–0.128) | 79.6% (74.6%–84.9%) |
<!-- /table -->

## S9. Loss of sensitivity at the inpatient threshold

Table S9. Composite sensitivity among outpatients at the inpatient threshold, with its lower Wilson bound, and the loss from 90% with its upper bound (section 3.7).

<!-- table:a1 -->
| Arm | Outpatient sensitivity | Loss, points | Loss at most 9 points |
|---|---|---|---|
| ResNet trained on EchoNext | 71.6% (low 65.9%) | 18.4 (up to 24.1) | no |
| ResNet at random initialisation, probes | 78.6% (low 73.3%) | 11.4 (up to 16.7) | no |
| ECGFounder frozen, probes | 71.6% (low 65.9%) | 18.4 (up to 24.1) | no |
| EchoNext mini-model | 72.7% (low 67.1%) | 17.3 (up to 22.9) | no |
| ECGFounder fine-tuned, seed 0 | 72.7% (low 67.1%) | 17.3 (up to 22.9) | no |
| ECGFounder fine-tuned, seed 1 | 72.0% (low 66.3%) | 18.0 (up to 23.7) | no |
| ECGFounder fine-tuned, seed 2 | 75.6% (low 70.2%) | 14.4 (up to 19.8) | no |
<!-- /table -->

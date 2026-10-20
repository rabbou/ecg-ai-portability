# A model's PPV, recomputed for a new hospital

The measurement of the positive predictive value recomputed by Bayes' theorem against the one observed, across transfers between hospitals, care settings and corpora, is part of the article [REPORT.md](REPORT.md): sections 2.5 and 3.1 for the measurement, 3.5 for the predictions that need no diagnosis, and 3.6 for the three repairs judged on net benefit. The methods in statistical terms are in Part 3 of [SUPPLEMENT.md](SUPPLEMENT.md).

## Repairs of the announced PPV

### Objective

At a fixed threshold, a model's PPV depends on prevalence through Bayes' theorem, so a PPV that holds from one site to the next requires a threshold that moves with the site. Five repairs a receiving hospital can apply were compared on the study's scores, by the absolute difference between the PPV each repair announces and the PPV observed in target patients it did not use, as a function of the number of local labels n. The PPV is the primary quantity because it is the figure a hospital is given; discrimination (AUROC) and net benefit at a 10% threshold are reported per transfer in [`results/ppv_repairs.json`](results/ppv_repairs.json).

### Methods

Three receiving hospitals carry the comparison. At Columbia, the four EchoNext models are calibrated on inpatients and applied to emergency and outpatient ECGs (76 transfers). At Beth Israel Deaconess, MIMIC-IV-ECG is the target of the seven-source rotation, with machine-read labels, so its observed PPV is agreement with the cart's statement (28 transfers). At Chongqing, the PTB-XL infarction model is applied to the hospital's ECGs (1 transfer). Transfers within a hospital are not independent, and the conclusion rests on three sites. Shandong, the other rotation corpora, CODE-15%, the CODE test set and ZZU are a secondary reserve in [`results/ppv_repairs_reserve.json`](results/ppv_repairs_reserve.json); pairs whose target is the source population are in-distribution controls.

For each transfer, the target v is the PPV observed at the source at the source's threshold of 90% sensitivity. It is computed from the source arrays alone, in a first pass before any repair runs, and the file records the digest of that table. The five repairs:

1. prevalence estimated without labels by EM (Saerens et al., 2002), then Bayes' theorem at the source threshold;
2. logistic recalibration on n labels, announcing the mean recalibrated probability of the flagged patients;
3. a threshold refitted for 90% sensitivity on one half of the n labels, announcing the PPV read on the other half, with its one-sided Clopper-Pearson bound;
4. the lowest threshold of a grid fixed by the source scores at which an exact binomial test rejects PPV ≤ v at δ = 0.1, tested in fixed sequence from the highest threshold down (Learn-then-Test, Angelopoulos et al.), announcing v;
5. weighted conformal selection (Jin and Candès, arXiv:2307.09291, Algorithm 1 with deterministic pruning) at a false discovery rate q = 1 − v, without labels, with covariate-shift weights estimated from the model's score.

Labels are drawn by patient, n = 25, 50, 100, 200 and 400, 200 draws per budget, and the remaining target patients form the evaluation set; repairs 1 and 5 use the whole target.

### Results

Without local labels, the prevalence correction announced the PPV with a median absolute error of 13.1 points at Columbia, 11.4 points at Beth Israel and 51.7 points at Chongqing. It announced less than was observed in every Columbia transfer and more than was observed at Chongqing. The published label-free figure for Columbia, over that study's selection of cells, was 12.6 points. In the in-distribution controls the correction erred by 2.4 points; given the true prevalence instead of the estimated one, Bayes' theorem erred there by 0.5 points, against 0.2 points over the published study's EchoNext controls.

Local labels reduced the error. With 25 labels, the recalibrated PPV erred by 5.8 points at Columbia, 8.5 points at Beth Israel and 9.3 points at Chongqing, and the PPV read on a half-sample at a refitted threshold by 6.0, 5.7 and 6.6 points. With 100 labels the two errors were 3.1 and 2.6 points at Columbia, 5.7 and 2.2 points at Beth Israel, and 3.8 and 3.5 points at Chongqing. With 400 labels they were 1.9 and 1.6 points, 3.5 and 1.7 points, and 1.8 and 2.6 points. From 100 labels on, the errors fell on both sides of the observed PPV in about equal shares, with 0.44 to 0.55 of the draws over-announcing; with fewer labels, recalibration over-announced in up to 0.72 of the draws.

The certified threshold kept its guarantee but was seldom found. At n = 100, the share of draws whose observed PPV fell below v was 0.022 at Columbia, 0.014 at Beth Israel and 0.010 at Chongqing, all under δ. A threshold was certified in 0.168, 0.251 and 0.010 of the draws: at most targets, no threshold of the grid reaches the source's PPV.

Weighted conformal selection did not control the false discovery proportion at Chongqing, where its mean was 0.75 for a level of 0.39. At Columbia it was 0.64 for a level of 0.77, and at Beth Israel 0.46 for 0.60.

### Falsifiers

The label-free correction would have sufficed outside Columbia if the upper bound of the bootstrap interval of its median absolute error had stayed under two points at both Beth Israel and Chongqing. The bounds were 30.3 points and 52.8 points, so the need for local labels is not restricted to Columbia. Chongqing contributes a single transfer, so its bound reflects the resampling of its patients only.

Certification would have failed under real draws if its violation rate at n = 100 had exceeded 0.15 in a hospital; it did not. The budget would not suffice to certify v if a threshold existed in fewer than half of the draws, and that holds at all three hospitals: 100 labels do not certify the source's PPV at these targets.

### Limitations

The comparison rests on three hospitals, with correlated transfers within each. Beth Israel's outcome is a machine statement. Each draw's observed PPV is estimated on a finite evaluation set, which adds sampling noise to the measured error and to the violation rate. Repair 5 uses the model's score as its only covariate, so it assumes that the outcome depends on the ECG only through that score; the false discovery proportion at Chongqing suggests the assumption fails there. The target v is the source's PPV, and a hospital that sets another target would see other rates of certification. Calibration curves are not redrawn here; the calibration of the recalibrated model at 100 labels is in [`results/repairs.json`](results/repairs.json).

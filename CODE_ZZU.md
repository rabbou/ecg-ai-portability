# Conformal coverage of four ECG diagnoses in Brazilian telehealth and in children

Thresholds calibrated at one of six hospitals and carried to the Telehealth Network of Minas Gerais miss 90% coverage of the cases in 56 of 80 source, target and diagnosis cells, 46 of them with the whole 95% interval below 90%. On the CODE test set, read by two cardiologists, the six foreign sources cover between 46.4% and 100% of first-degree block and between 50.0% and 100% of left bundle-branch block. Carried to children at Zhengzhou, the same thresholds miss 90% in 24 of 28 cells. A model trained on CODE-15% under the study's protocol does not learn: its validation macro AUROC is 0.571, and its tracings carry DC offsets two orders of magnitude larger than those of the other corpora.

## Three corpora outside the hospitals of the rotation

The rotation of the [README](README.md) and of [MIMIC.md](MIMIC.md) holds six hospital corpora. Three open releases add a care setting and a population it lacks. CODE-15% (Zenodo 4916206, CC BY 4.0) holds 345,779 exams of 233,770 patients read by the Telehealth Network of Minas Gerais from 2010 to 2016; its labels are reconciled from the telehealth cardiologist's report, the Uni-G statements and Minnesota codes (Ribeiro et al., Nat Commun 2020). The CODE test set (Zenodo 3765780, CC BY 4.0) holds 827 exams of 827 patients from 2018, each read by two cardiologists, a third settling disagreements. ZZU pECG (figshare 27078763, CC BY 4.0; Tan et al., Sci Data 2025) holds 14,190 ECGs of 11,643 inpatient children aged 0 to 14 at the First Affiliated Hospital of Zhengzhou University; it is a side arm and enters no adult summary.

## From each release to the rotation's diagnoses

`src/ecs/code_ecg.py` and `src/ecs/zzu.py` map each release onto atrial fibrillation, left and right bundle-branch block and first-degree atrioventricular block. Sinus rhythm is refused on all three, as on Shandong: CODE carries only an automatic "normal ECG", and ZZU has no sinus-rhythm statement. ZZU's AHA codes go through Leinonen's AHA-to-SNOMED table, the bridge already used for Shandong; its 165 records of first-degree block written under the Chinese standard, which has no AHA code, are counted with the 12 coded "prolonged PR interval". `results/code_zzu_label_map.json` holds every count and every ambiguity.

| Corpus | AF | LBBB | RBBB | First-degree block | Reference |
|---|---|---|---|---|---|
| CODE test set, 827 exams | 13 | 30 | 34 | 28 | Ribeiro et al. 2020, Table 1: 13, 30, 34, 28 |
| ZZU pECG, statements over 14,190 records | 31 | 31 | 395 | 165 + 12 | Tan et al. 2025, Supplementary Table 1: 31, 31, 395, 165 + 12 |
| CODE-15%, share of 345,779 exams | 2.03% | 1.74% | 2.80% | 1.65% | full CODE, Ribeiro et al. 2020, Table 1: 1.8%, 1.7%, 2.7%, 1.5% |

The counts reproduce both publishers' tables exactly. CODE-15% publishes no table of its own; its shares sit within 0.25 points of those of the full CODE dataset it was sampled from.

## The sample of CODE-15%, and the test set kept whole

CODE-15% ships its exams as 44 GB of zipped HDF5 at 400 Hz, padded with zeros to 4,096 samples. Only 132,161 exams hold ten seconds; 201,084 hold 7.3 seconds. `scripts/code15_extract.py` reads every exam's span on the machine that holds the release, and the sample is drawn among the 92,246 patients with a ten-second exam: one such exam per patient (seed 27), then 22,000 patients stratified on the combination of the four flags, which keeps each flag's prevalence within 0.0004 of the pool's. The ten-second pool carries less right bundle-branch block than the whole release (2.1% against 2.8%). The test set is kept whole: 539 of its exams last 7.3 to 7.8 seconds and are zero-filled to ten, and its ten-second exams are also reported alone as `code_test_10s`. ZZU keeps one twelve-lead record of at least ten seconds per child, capped at the rotation's 8,000 test records.

## The rotation with seven sources

The six sources of MIMIC.md and CODE-15% are trained with the same seed, caps and schedule under `results/rotation_seven` on an Apple M5 with the `mps` backend. The six retrained models reproduce the scores of `results/rotation_six` exactly: on the 169 cells the two runs share, the largest difference in coverage is 0.0. Each source's per-label thresholds are fitted once, on its whole calibration part; `results/code_zzu_rotation.csv` gives every pair, diagnosis and rule with 95% Wilson intervals.

| Calibrated at | AF | LBBB | RBBB | First-degree block |
|---|---|---|---|---|
| PTB-XL | 61.5% | 50.0% | 50.0% | 46.4% |
| Shandong | 100.0% | 93.3% | 91.2% | 53.6% |
| Chapman-Shaoxing and Ningbo | 53.8% | 100.0% | 64.7% | 67.9% |
| Georgia | 100.0% | 93.3% | 85.3% | 92.9% |
| CPSC | 53.8% | 100.0% | 67.6% | 60.7% |
| Beth Israel | 69.2% | 93.3% | 82.4% | 100.0% |
| CODE-15% | 38.5% | 80.0% | 29.4% | 75.0% |

Coverage of the cases in the CODE test set, 90% asked for, per-label calibration; 13, 30, 34 and 28 cases.

The discrimination of the foreign models on the test set is lower than at any hospital of the rotation: the median AUROC over the six sources is 0.817 for atrial fibrillation, 0.892 for left and 0.865 for right bundle-branch block, and 0.833 for first-degree block. With 13 to 34 cases per diagnosis, each figure carries a wide interval, given per cell in the grid.

The tracings explain part of it. Read as the models read them, unfiltered, the CODE-15% tracings have a median absolute offset of 0.76 mV per lead against 0.003 mV in PTB-XL, and 39% of their leads sit more than 1 mV from zero; their QRS range in lead II, band-passed at 1 to 40 Hz, is about 1.5 times that of the adult hospital corpora (`amplitude` in `results/code_zzu_label_map.json`). The study's protocol applies no filter and no per-record normalisation, so this difference reaches the network as a difference between hospitals. Under it, the CODE-15% model stops at epoch 1 with a validation macro AUROC of 0.571, and the foreign models read CODE-15% at a median AUROC between 0.548 and 0.634. The CODE-15% rows of the grid are therefore a measure of the protocol's limit on this release, not of transfer: at home its thresholds cover 88.9% to 95.5% of the cases by sending 77% to 82% of tracings to a human reader, and carried elsewhere they miss 90% in 30 of 32 cells.

## Children at Zhengzhou, a side arm

| Calibrated at | AF | LBBB | RBBB | First-degree block |
|---|---|---|---|---|
| PTB-XL | 33.3% | 33.3% | 71.3% | 65.6% |
| Shandong | 46.7% | 83.3% | 94.6% | 65.6% |
| Chapman-Shaoxing and Ningbo | 13.3% | 100.0% | 86.0% | 70.5% |
| Georgia | 93.3% | 50.0% | 89.9% | 68.8% |
| CPSC | 26.7% | 66.7% | 96.9% | 59.0% |
| Beth Israel | 53.3% | 50.0% | 80.6% | 65.6% |

Coverage of the cases in ZZU pECG, 90% asked for, per-label calibration; 15, 6, 129 and 61 cases.

The adult models discriminate right bundle-branch block in children as well as in adults (median AUROC 0.966) and first-degree block nearly as well (0.927), yet first-degree block is covered at 59.0% to 70.5% from every source. This suggests that the scores of children fall on the other side of thresholds fitted on adults, rather than that the models fail to rank them. Atrial fibrillation and left bundle-branch block hold 15 and 6 cases and are reported, not interpreted.

## Limits

CODE-15%'s labels are reconciled report labels and the test set's are a cardiologist consensus; the two are kept as separate corpora and never pooled. HuBERT-ECG and ECG-JEPA saw CODE-15% at pre-training (`ecs.encoders.SAW`); the rotation uses the study's ResNet trained from scratch, which saw nothing outside its source, and whether HuBERT-ECG also saw the CODE test set its paper does not say. The encoder arms are not extended to these corpora. A high-pass or per-record normalised variant of the protocol, applied to every corpus, would say how much of the CODE result is offset and how much is population; it is not run here.

## Reproduce

```bash
# on the machine holding CODE-15%'s zips (numpy and h5py only)
uv run --no-project --with h5py --with numpy python scripts/code15_extract.py spans \
  --zips ~/data/code15 --work ~/tmp-code15 --out ~/tmp-code15/spans.csv          # 80 min
uv run python scripts/code_zzu_label_table.py --spans spans.csv --ids-out ids.txt --no-rotation
uv run --no-project --with h5py --with numpy python scripts/code15_extract.py extract \
  --zips ~/data/code15 --work ~/tmp-code15 --spans spans.csv --ids ids.txt \
  --out ~/data/code15-sample/code15_sample.h5                                     # 80 min
for s in ptbxl sph chapman_ningbo georgia cpsc mimic code15; do
  uv run python scripts/train_source.py --source $s --device mps --rotation-dir results/rotation_seven
done                                                                              # 4 min each
uv run python scripts/score_rotation.py --sources ptbxl,sph,chapman_ningbo,georgia,cpsc,mimic,code15 \
  --corpora ptbxl,sph,chapman_ningbo,georgia,cpsc,mimic,code15,code_test,zzu \
  --device mps --rotation-dir results/rotation_seven
uv run python scripts/code_zzu_label_table.py --spans ~/data/code15-sample/spans.csv
uv run python scripts/code_zzu_rotation.py
```

ZZU's records are unpacked from its split archive with `zip -s 0 Child_ecg.zip --out full.zip` and `unzip full.zip -d ~/data/zzu-pecg-records`. `tests/test_code_zzu_labels.py` holds the label table to the publishers' counts, and `tests/test_code_zzu_results.py` recomputes every figure of this document from the committed scores.

CODE-15% and the CODE test set are Ribeiro et al., Nat Commun 11, 1760 (2020), doi:10.1038/s41467-020-15432-4; ZZU pECG is Tan et al., Sci Data 2025, doi:10.1038/s41597-025-05225-z; all three under CC BY 4.0. The committed score files hold one row per record (identifier, labels, probabilities), derived data under the same licence.

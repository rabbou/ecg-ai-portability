# Results

Every number and figure the article prints is redrawn from the files in `results/`. Each file below is written by one script in `scripts/`; the first line of that script's docstring says what it measures. `tests/test_provenance.py` pins each file to the commit that holds its code.

## Files by script

| File | Written by | What it holds |
|---|---|---|
| `abstention.json` | `scripts/abstention_table.py` | How often the model declines to answer, and at what confidence |
| `arms.json` | `scripts/arms_table.py` | Four encoder arms compared on the same calibration-transfer break |
| `auxiliary.json` | `scripts/auxiliary.py` | The quantities the discussion and the limitations rest on |
| `baseline.json` | summary of the PTB-XL baseline, read by `scripts/figures.py` and `tests/test_baseline.py` | The supervised baseline's headline numbers; the run's own files are in `baseline/` |
| `duplicate_groups.json` | `scripts/duplicate_scan.py` | Which tracings a corpus holds twice |
| `echonext_clinical.json` | `scripts/echonext_clinical.py` | EchoNext counted in patients: what the inpatient threshold does to 100 outpatients with SHD |
| `echonext_coverage.csv`, `echonext_transfer.json` | `scripts/echonext_transfer.py` | EchoNext inpatients to outpatients: the coverage table and the data behind the transfer reports |
| `echonext_ppv_gap.json`, `ppv_gap.json`, `ppv_gap.csv` | `scripts/ppv_gap.py` | The PPV recomputed by Bayes' rule against the PPV observed, per transfer |
| `echonext_provenance.json` | `scripts/echonext_provenance.py` | One provenance row per EchoNext tracing |
| `echonext_severity.json` | `scripts/echonext_severity.py` | Whether the outpatients' lost coverage is a matter of milder disease |
| `infarction_sites.json` | `scripts/infarction_sites.py` | Infarction at the three sites: separation and threshold movement |
| `ingest_report.json` | `scripts/scan_corpora.py` | Every record of every corpus through the ingestion chain |
| `label_map.json` | `scripts/label_table.py` | What each of the five corpora calls each of the five diagnoses |
| `outcomes.json` | `scripts/outcomes.py` | What a patient of each label gets under three ways of thresholding |
| `perturbations.json`, `perturbations.npz` | `scripts/perturbations.py` | What four acquisition faults do to coverage |
| `ppv_intervals.json` | `scripts/ppv_intervals.py` | 95% intervals for the shares the article quotes that no other file carries |
| `repairs.json` | `scripts/repairs.py` | Three repairs on the same scores, judged on net benefit |
| `rotation.json`, `rotation.csv` | `scripts/rotation_table.py` | The source rotation: five hospitals, each calibrating once and spending everywhere |
| `rotation_uncertainty.json`, `rotation_uncertainty.csv` | `scripts/rotation_uncertainty.py` | Three things the rotation table cannot say on its own |
| `seen_target.json` | `scripts/seen_target_count.py` | How much infarction the PhysioNet 2021 bundle holds outside PTB-XL |
| `shift.json` | `scripts/shift_table.py` | What the coverage guarantee is worth at another hospital |
| `split_leak.json` | `scripts/split_leak.py` | What the splits leaked, and what they leak once repeats are joined |
| `subgroups.json` | `scripts/subgroups.py` | Coverage inside sex and age |
| `target_scale.json` | `scripts/target_scale.py` | How many labelled records a hospital needs before the guarantee holds there |
| `timing.json` | `scripts/timing_probe.py` | Seconds per hundred PTB-XL records through each encoder arm |
| `timing_esprimo.json` | `scripts/timing_probe.py`, run on a second machine | The same measurement on that machine, kept as a single moment |

## Folders

| Folder | What it holds | Written by |
|---|---|---|
| `baseline/` | The PTB-XL baseline run: config, metrics, scores | `scripts/train_baseline.py` |
| `embeddings/<arm>/<corpus>.json` | One encoder arm over one corpus | `scripts/extract_embeddings.py` |
| `external/` | The PTB-XL model spent on Shandong (`sph`) and Chongqing (`acs`) | `scripts/score_external.py` |
| `figures/` | The article's figures, English and French (`_fr`) | `scripts/figures.py`, `scripts/paper_figures.py`, `scripts/ppv_figures.py`, `scripts/bilingual_figures.py` |
| `rotation/<source>/` | One model per source corpus: config, metrics, scores | `scripts/train_source.py`, `scripts/score_rotation.py` |

# Transfer reports

One page per EchoNext model for the transfer from Columbia inpatients to Columbia outpatients, with the PPV gap for every finding. Each page is written by `scripts/echonext_transfer.py` from `results/echonext_transfer.json`, and `tests/test_paper_numbers.py` checks the numbers on it. The article's section 3.7 supersedes the pages' threshold ladder.

| Page | Model |
|---|---|
| [resnet_inpatient_to_outpatient.md](resnet_inpatient_to_outpatient.md) | The study's residual network, trained on EchoNext |
| [echonext_mini_inpatient_to_outpatient.md](echonext_mini_inpatient_to_outpatient.md) | The published EchoNext mini-model, published weights |
| [ecgfounder_inpatient_to_outpatient.md](ecgfounder_inpatient_to_outpatient.md) | ECGFounder, frozen, logistic regressions |
| [random_init_inpatient_to_outpatient.md](random_init_inpatient_to_outpatient.md) | Random initialisation, frozen, logistic regressions |

"""Study P-A on EchoNext: weighted coverage (A2), calibration and abstention (A3), A1's verdict.

Calibration is the inpatient ECGs of the validation split, the targets the
inpatient, emergency and outpatient ECGs of the test split, as in
``scripts/echonext_transfer.py``.  Per arm, label and target context:

- coverage of cases and non-cases for the study's single threshold (``plain``),
  the per-label conformal sets (``perlabel``), the same weighted by w(x) on
  each covariate set (``weighted``), and per-label thresholds refitted on n
  local labels (``local_labels``, the study's ladder);
- slope, intercept, ICI, Brier, AUGRC and the sensitivity kept at 80% coverage,
  each with a bootstrap interval.

The weights are fitted once per target context and covariate set and shared by
every arm, so arms differ by their scores alone.  The embedding covariate is the
frozen ECGFounder's 1024-d vector, filed in the embedding store.

An arm whose scores are not under ``$ECS_ECHONEXT_DERIVED/scores`` is left out
and named in ``results/structure.json``; the fine-tuned arms appear once
``scripts/echonext_finetune.py`` has filed them.

Writes ``results/structure_coverage.csv``, ``results/structure_calibration.json``
and ``results/structure.json``.

``--pages`` re-renders the tables of STRUCTURE.md alone from the committed
result files, between its ``<!-- table:<name> -->`` and ``<!-- /table -->`` lines.

Usage: .venv/bin/python scripts/structure.py [--pages]
"""

from __future__ import annotations

import argparse
import json
import re
import time
from typing import Any

import numpy as np
import pandas as pd
from echonext_scores import ECGFOUNDER_VERSION, Tabular
from echonext_transfer import SCORES_DIR, scores_for
from numpy.typing import NDArray

from ecs.config import REPO_ROOT, RESULTS_DIR
from ecs.conformal import conformal_quantile
from ecs.echonext import (
    COMPOSITE,
    DERIVED_DIR,
    DISTRIBUTION,
    LABELS,
    read_metadata,
    transfer_cohorts,
)
from ecs.embedding_store import EmbeddingStore
from ecs.metrics import effective_sample_size
from ecs.provenance import provenance_block
from ecs.structure import (
    JUDGED_MIN_CASES,
    a2_verdict,
    calibration_cell,
    domain_weights,
    effective_cases,
    weighted_perlabel_sets,
)
from ecs.structure_report import load, tables
from ecs.transfer import ALPHA, conformal_sets, coverage_row, ladder_rows

ARMS = (
    "resnet",
    "random_init",
    "ecgfounder",
    "echonext_mini",
    "ecgfounder_ft_s0",
    "ecgfounder_ft_s1",
    "ecgfounder_ft_s2",
)
FINETUNED = ("ecgfounder_ft_s0", "ecgfounder_ft_s1", "ecgfounder_ft_s2")
TARGETS = ("inpatient", "emergency", "outpatient")
COVARIATES = ("age_sex", "tabular", "embedding")
VERDICT_CONTEXT = "outpatient"
VERDICT_COVARIATES = "embedding"
# A1: the published loss of composite sensitivity, inpatient threshold to outpatients,
# for the study's ResNet (90% against 71.6%), and the bound the protocol sets on it.
TARGET_SENSITIVITY = 1 - ALPHA
A1_MAX_LOSS = 0.09

Positions = NDArray[np.int_]
Covariates = dict[str, NDArray[np.float64]]


def covariates(meta: pd.DataFrame, positions: Positions) -> Covariates:
    """Each covariate set for ``positions``, rows in the same order."""
    part = meta.iloc[positions]
    provenance = pd.read_csv(
        DERIVED_DIR / "provenance.csv.gz", usecols=["ecg_key", "tracing_sha256"]
    )
    digest_of = dict(zip(provenance["ecg_key"], provenance["tracing_sha256"], strict=True))
    store = EmbeddingStore("ecgfounder", ECGFOUNDER_VERSION)
    digests = [digest_of[k] for k in part["ecg_key"]]

    def refuse(missing: Positions) -> NDArray[np.float32]:
        raise RuntimeError(f"{len(missing)} tracings have no filed ECGFounder vector")

    return {
        "age_sex": np.column_stack(
            [part["age_at_ecg"].to_numpy(float), (part["sex"] == "male").to_numpy(float)]
        ),
        "tabular": Tabular(meta)(positions).astype(np.float64),
        "embedding": store.embed(digests, refuse).astype(np.float64),
    }


def available_arms() -> tuple[list[str], list[str]]:
    present = [a for a in ARMS if (SCORES_DIR / f"{a}.npz").exists()]
    return present, [a for a in ARMS if a not in present]


def plain_threshold(p_cal: NDArray[np.float64], y_cal: NDArray[np.int_]) -> float:
    """The probability at which 90% of calibration cases are flagged, as ``plain`` sets it."""
    return 1.0 - conformal_quantile(1.0 - p_cal[y_cal == 1], ALPHA)


def coverage_rows(
    arm: str,
    context: str,
    k: int,
    cal: tuple[NDArray[np.float64], NDArray[np.int_]],
    tgt: tuple[NDArray[np.float64], NDArray[np.int_]],
    weights: dict[str, Any],
    patients: pd.Series,
) -> list[dict[str, Any]]:
    (p_cal, y_cal), (p_tgt, y_tgt) = cal, tgt
    head = {"arm": arm, "label": LABELS[k], "context": context}
    sets = conformal_sets(p_cal, y_cal, p_tgt, ALPHA)
    rows = [
        {**head, "method": m, "interval": "wilson", **coverage_row(sets[m], y_tgt)}
        for m in ("plain", "perlabel")
    ]
    for name, w in weights.items():
        weighted = weighted_perlabel_sets(p_cal, y_cal, w.calibration, p_tgt, w.target, ALPHA)
        case_weights = w.calibration[y_cal == 1]
        rows.append(
            {
                **head,
                "method": "weighted",
                "covariates": name,
                "interval": "wilson",
                **coverage_row(weighted, y_tgt),
                "ess_calibration_cases": effective_sample_size(case_weights),
                "n_effective": effective_cases(int(y_tgt.sum()), case_weights),
                "domain_auroc": w.domain_auroc,
                "weight_cap": w.cap,
            }
        )
    for rung in ladder_rows(p_cal, y_cal, p_tgt, y_tgt, patients):
        if rung["labels"] == 0:
            continue
        rows.append(
            {
                **head,
                "method": "local_labels",
                "n_labels": rung["labels"],
                "interval": f"p10-p90 of {rung['draws']} draws",
                "n": rung["n_eval"],
                "n_pos": rung["n_eval_pos"],
                "coverage_pos": rung["coverage_pos_mean"],
                "coverage_pos_low": rung["coverage_pos_p10"],
                "coverage_pos_high": rung["coverage_pos_p90"],
                "coverage_neg": rung["coverage_neg_mean"],
            }
        )
    return rows


def measure(
    meta: pd.DataFrame, arms: list[str], weights: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cohorts = transfer_cohorts(meta, "inpatient", TARGETS)
    labels = meta[list(LABELS)].to_numpy(dtype=int)
    coverage: list[dict[str, Any]] = []
    calibration: dict[str, Any] = {}
    for arm in arms:
        probs = scores_for(arm, meta)[0].astype(np.float64)
        calibration[arm] = {}
        for context, rows in cohorts.targets.items():
            calibration[arm][context] = {}
            for k, label in enumerate(LABELS):
                cal = probs[cohorts.calibration, k], labels[cohorts.calibration, k]
                tgt = probs[rows, k], labels[rows, k]
                patients = meta["patient_key"].iloc[rows]
                coverage += coverage_rows(arm, context, k, cal, tgt, weights[context], patients)
                threshold = plain_threshold(*cal)
                calibration[arm][context][label] = calibration_cell(tgt[1], tgt[0], threshold)
        print(f"{arm}: measured", flush=True)
    return coverage, calibration


def judged_labels(meta: pd.DataFrame, rows: Positions) -> list[str]:
    """The anomalies with at least JUDGED_MIN_CASES target cases, composite apart."""
    counts = meta[list(LABELS)].iloc[rows].sum()
    return [label for label in LABELS[:-1] if counts[label] >= JUDGED_MIN_CASES]


def _row(rows: pd.DataFrame, label: str) -> dict[str, Any]:
    return {str(k): v for k, v in rows.loc[label].items()}


def verdicts(table: pd.DataFrame, arms: list[str], judged: list[str]) -> dict[str, Any]:
    """A2 per arm and covariate set on the outpatients, and A1 for each fine-tuned seed."""
    out: dict[str, Any] = {"a2": {}, "a1": {}}
    target = table[table["context"] == VERDICT_CONTEXT]
    for arm in arms:
        out["a2"][arm] = {}
        for name in COVARIATES:
            rows = target[
                (target["arm"] == arm)
                & (target["method"] == "weighted")
                & (target["covariates"] == name)
            ].set_index("label")
            out["a2"][arm][name] = a2_verdict(
                _row(rows, COMPOSITE), {k: _row(rows, k) for k in judged}
            )
    plain = target[(target["method"] == "plain") & (target["label"] == COMPOSITE)].set_index("arm")
    for arm in arms:
        row = plain.loc[arm]
        out["a1"][arm] = {
            "sensitivity": row["coverage_pos"],
            "sensitivity_low": row["coverage_pos_low"],
            "loss": TARGET_SENSITIVITY - row["coverage_pos"],
            "loss_high": TARGET_SENSITIVITY - row["coverage_pos_low"],
            "halved": bool(TARGET_SENSITIVITY - row["coverage_pos_low"] <= A1_MAX_LOSS),
        }
    seeds = [a for a in FINETUNED if a in arms]
    out["a1_finetuned_halves_the_loss"] = (
        all(out["a1"][a]["halved"] for a in seeds) if len(seeds) == len(FINETUNED) else None
    )
    return out


# The script, and every module whose behaviour the numbers depend on.
PRODUCERS = [
    "scripts/echonext_scores.py",
    "scripts/echonext_transfer.py",
    "scripts/structure.py",
    "src/ecs/calibration.py",
    "src/ecs/conformal.py",
    "src/ecs/echonext.py",
    "src/ecs/embedding_store.py",
    "src/ecs/metrics.py",
    "src/ecs/splits.py",
    "src/ecs/structure.py",
    "src/ecs/transfer.py",
]
DOC = REPO_ROOT / "STRUCTURE.md"
BLOCK = re.compile(r"(<!-- table:(\w+) -->\n)(.*?)(<!-- /table -->)", re.DOTALL)


def write_pages() -> None:
    """Replace each table block of STRUCTURE.md with its rendering from the result files."""
    rendered = tables(*load())
    text = DOC.read_text()
    names = [m.group(2) for m in BLOCK.finditer(text)]
    if sorted(names) != sorted(rendered):
        raise ValueError(f"STRUCTURE.md holds blocks {names}, the renderer {sorted(rendered)}")
    DOC.write_text(BLOCK.sub(lambda m: f"{m.group(1)}{rendered[m.group(2)]}\n{m.group(4)}", text))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", action="store_true", help="re-render STRUCTURE.md only")
    if parser.parse_args().pages:
        write_pages()
        return
    started = time.time()
    meta = read_metadata()
    cohorts = transfer_cohorts(meta, "inpatient", TARGETS)
    arms, absent = available_arms()
    source = covariates(meta, cohorts.calibration)
    weights: dict[str, dict[str, Any]] = {}
    for context, rows in cohorts.targets.items():
        target = covariates(meta, rows)
        weights[context] = {name: domain_weights(source[name], target[name]) for name in COVARIATES}
        print(f"{context}: weights fitted", flush=True)
    coverage, calibration = measure(meta, arms, weights)
    table = pd.DataFrame(coverage)
    judged = judged_labels(meta, cohorts.targets[VERDICT_CONTEXT])
    summary = {
        "distribution": DISTRIBUTION,
        "alpha": ALPHA,
        "arms": arms,
        "arms_absent": absent,
        "calibration_n": len(cohorts.calibration),
        "targets_n": {c: len(r) for c, r in cohorts.targets.items()},
        "judged_labels": judged,
        "verdict_context": VERDICT_CONTEXT,
        "verdict_covariates": VERDICT_COVARIATES,
        "domain": {
            c: {n: {"domain_auroc": w.domain_auroc, "cap": w.cap} for n, w in ws.items()}
            for c, ws in weights.items()
        },
        **verdicts(table, arms, judged),
        "seconds": round(time.time() - started, 1),
        "provenance": provenance_block(PRODUCERS),
    }
    table.to_csv(RESULTS_DIR / "structure_coverage.csv", index=False)
    (RESULTS_DIR / "structure_calibration.json").write_text(
        json.dumps(calibration, indent=2) + "\n"
    )
    (RESULTS_DIR / "structure.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_pages()


if __name__ == "__main__":
    main()

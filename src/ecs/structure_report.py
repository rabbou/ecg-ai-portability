"""The tables of STRUCTURE.md, rendered from the three committed result files of study P-A.

STRUCTURE.md prints each table exactly as ``tables`` renders it, and
tests/test_structure_results.py holds the document to that: a figure changes in
the document only by re-running ``scripts/structure.py`` and re-rendering.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import RESULTS_DIR
from .echonext import COMPOSITE

__all__ = ["ARM_NAMES", "LABEL_NAMES", "load", "pct", "tables"]

ARM_NAMES = {
    "resnet": "ResNet trained on EchoNext",
    "random_init": "ResNet at random initialisation, probes",
    "ecgfounder": "ECGFounder frozen, probes",
    "echonext_mini": "EchoNext mini-model",
    "ecgfounder_ft_s0": "ECGFounder fine-tuned, seed 0",
    "ecgfounder_ft_s1": "ECGFounder fine-tuned, seed 1",
    "ecgfounder_ft_s2": "ECGFounder fine-tuned, seed 2",
}
LABEL_NAMES = {
    COMPOSITE: "Composite",
    "lvef_lte_45_flag": "LVEF ≤ 45%",
    "lvwt_gte_13_flag": "LV wall ≥ 13 mm",
    "aortic_stenosis_moderate_or_greater_flag": "Aortic stenosis",
    "pasp_gte_45_flag": "PASP ≥ 45 mmHg",
}
COVARIATE_NAMES = {"age_sex": "age, sex", "tabular": "ECG measurements", "embedding": "embedding"}
CONTEXT = "outpatient"


def pct(x: float | None) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{100 * x:.1f}%"


def _interval(row: pd.Series) -> str:
    low, high = pct(row["coverage_pos_low"]), pct(row["coverage_pos_high"])
    return f"{pct(row['coverage_pos'])} ({low}–{high})"


def load(root: Path = RESULTS_DIR) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    summary = json.loads((root / "structure.json").read_text())
    coverage = pd.read_csv(root / "structure_coverage.csv")
    calibration = json.loads((root / "structure_calibration.json").read_text())
    return summary, coverage, calibration


def _cell(coverage: pd.DataFrame, **where: Any) -> pd.Series:
    mask = pd.Series(True, index=coverage.index)
    for key, value in where.items():
        mask &= coverage[key] == value
    rows = coverage[mask]
    if len(rows) != 1:
        raise KeyError(f"{len(rows)} rows for {where}")
    return rows.iloc[0]


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def domain_table(summary: dict[str, Any]) -> str:
    rows = [
        [
            context,
            *(f"{summary['domain'][context][c]['domain_auroc']:.3f}" for c in COVARIATE_NAMES),
        ]
        for context in summary["domain"]
    ]
    return _table(["Target", *COVARIATE_NAMES.values()], rows)


def composite_table(summary: dict[str, Any], coverage: pd.DataFrame) -> str:
    """A2 on the composite: the single threshold and each weighting, outpatients."""
    rows = []
    for arm in summary["arms"]:
        plain = _cell(coverage, arm=arm, label=COMPOSITE, context=CONTEXT, method="plain")
        row = [ARM_NAMES[arm], _interval(plain)]
        for name in COVARIATE_NAMES:
            w = _cell(
                coverage,
                arm=arm,
                label=COMPOSITE,
                context=CONTEXT,
                method="weighted",
                covariates=name,
            )
            low, high = summary["a2"][arm][name]["composite_interval"]
            row.append(
                f"{pct(w['coverage_pos'])} ({pct(low)}–{pct(high)}, n={int(w['n_effective'])})"
            )
        rows.append(row)
    header = ["Arm", "Single threshold", *(f"Weighted on {v}" for v in COVARIATE_NAMES.values())]
    return _table(header, rows)


def anomaly_table(summary: dict[str, Any], coverage: pd.DataFrame) -> str:
    """A2 per judged anomaly, weighted on the embedding: sensitivity, size, Holm verdict."""
    rows = []
    for arm in summary["arms"]:
        verdict = summary["a2"][arm][summary["verdict_covariates"]]
        row = [ARM_NAMES[arm]]
        for label in summary["judged_labels"]:
            w = _cell(
                coverage,
                arm=arm,
                label=label,
                context=CONTEXT,
                method="weighted",
                covariates=summary["verdict_covariates"],
            )
            mark = "passes" if verdict["anomaly_pass"][label] else "fails"
            row.append(f"{pct(w['coverage_pos'])}, n={int(w['n_effective'])}, {mark}")
        row.append("holds" if verdict["repair_holds"] else "fails")
        rows.append(row)
    header = ["Arm", *(LABEL_NAMES[k] for k in summary["judged_labels"]), "Repair"]
    return _table(header, rows)


def ladder_table(summary: dict[str, Any], coverage: pd.DataFrame) -> str:
    """Composite sensitivity on outpatients with per-label thresholds refitted on n labels."""
    rungs = sorted(
        int(n) for n in coverage.loc[coverage["method"] == "local_labels", "n_labels"].unique()
    )
    rows = []
    for arm in summary["arms"]:
        row = [ARM_NAMES[arm]]
        for n in rungs:
            c = _cell(
                coverage,
                arm=arm,
                label=COMPOSITE,
                context=CONTEXT,
                method="local_labels",
                n_labels=n,
            )
            row.append(_interval(c))
        rows.append(row)
    return _table(["Arm", *(f"{n} labels" for n in rungs)], rows)


def calibration_table(summary: dict[str, Any], calibration: dict[str, Any]) -> str:
    """A3 on the composite, inpatient and outpatient targets."""
    rows = []
    for arm in summary["arms"]:
        for context in ("inpatient", CONTEXT):
            c = calibration[arm][context][COMPOSITE]

            def fig(key: str, digits: int = 2, c: dict[str, Any] = c) -> str:
                if c[key] is None:
                    return "n/a"
                low, high = c[key + "_low"], c[key + "_high"]
                return f"{c[key]:.{digits}f} ({low:.{digits}f}–{high:.{digits}f})"

            rows.append(
                [
                    ARM_NAMES[arm],
                    context,
                    fig("slope"),
                    fig("intercept"),
                    fig("ici", 3),
                    fig("brier", 3),
                    fig("augrc", 3),
                    f"{pct(c['sensitivity_at_80'])} ({pct(c['sensitivity_at_80_low'])}–"
                    f"{pct(c['sensitivity_at_80_high'])})",
                ]
            )
    header = ["Arm", "Target", "Slope", "Intercept", "ICI", "Brier", "AUGRC", "Sens. at 80% kept"]
    return _table(header, rows)


def a1_table(summary: dict[str, Any]) -> str:
    rows = [
        [
            ARM_NAMES[arm],
            f"{pct(a1['sensitivity'])} (low {pct(a1['sensitivity_low'])})",
            f"{100 * a1['loss']:.1f} (up to {100 * a1['loss_high']:.1f})",
            "yes" if a1["halved"] else "no",
        ]
        for arm, a1 in summary["a1"].items()
    ]
    header = ["Arm", "Outpatient sensitivity", "Loss, points", "Loss at most 9 points"]
    return _table(header, rows)


def tables(
    summary: dict[str, Any], coverage: pd.DataFrame, calibration: dict[str, Any]
) -> dict[str, str]:
    return {
        "domain": domain_table(summary),
        "composite": composite_table(summary, coverage),
        "anomalies": anomaly_table(summary, coverage),
        "ladder": ladder_table(summary, coverage),
        "calibration": calibration_table(summary, calibration),
        "a1": a1_table(summary),
    }

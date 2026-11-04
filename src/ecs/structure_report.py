"""The tables of STRUCTURE.md and its supplement, rendered from the committed results of P-A.

The two documents print each table exactly as ``tables`` renders it, and
tests/test_structure_results.py holds both to that: a figure changes in
a document only by re-running ``scripts/structure.py`` and re-rendering.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import RESULTS_DIR
from .echonext import COMPOSITE

__all__ = [
    "ARM_NAMES",
    "COVARIATE_NAMES",
    "LABEL_NAMES",
    "composite_points",
    "load",
    "pct",
    "tables",
]

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
    "aortic_regurgitation_moderate_or_greater_flag": "Aortic regurgitation",
    "mitral_regurgitation_moderate_or_greater_flag": "Mitral regurgitation",
    "tricuspid_regurgitation_moderate_or_greater_flag": "Tricuspid regurgitation",
    "pulmonary_regurgitation_moderate_or_greater_flag": "Pulmonary regurgitation",
    "rv_systolic_dysfunction_moderate_or_greater_flag": "RV dysfunction",
    "pericardial_effusion_moderate_large_flag": "Pericardial effusion",
    "pasp_gte_45_flag": "PASP ≥ 45 mmHg",
    "tr_max_gte_32_flag": "TR velocity ≥ 3.2 m/s",
}
COVARIATE_NAMES = {"age_sex": "age, sex", "tabular": "ECG measurements", "embedding": "embedding"}
CONTEXT = "outpatient"
EMERGENCY = "emergency"
PRIMARY_READING = "min_cases_kish"


def pct(x: float | None) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{100 * x:.1f}%"


def _interval(row: pd.Series) -> str:
    low, high = pct(row["coverage_pos_low"]), pct(row["coverage_pos_high"])
    return f"{pct(row['coverage_pos'])} ({low}–{high})"


def load(root: Path = RESULTS_DIR) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    summary = json.loads((root / "structure.json").read_text())
    summary["power"] = json.loads((root / "structure_power.json").read_text())["power"]
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


def composite_points(
    summary: dict[str, Any], coverage: pd.DataFrame, context: str
) -> dict[str, dict[str, tuple[float, float, float, int | None]]]:
    """Composite sensitivity, its Wilson bounds and the size they rest on, per arm and threshold.

    The inpatient threshold is judged on the target's cases; each weighting at the smaller
    of the cases and the Kish size, the size of the verdict's primary reading.
    """
    points: dict[str, dict[str, tuple[float, float, float, int | None]]] = {}
    for arm in summary["arms"]:
        plain = _cell(coverage, arm=arm, label=COMPOSITE, context=context, method="plain")
        points[arm] = {
            "plain": (
                plain["coverage_pos"],
                plain["coverage_pos_low"],
                plain["coverage_pos_high"],
                None,
            )
        }
        for name in COVARIATE_NAMES:
            w = _cell(
                coverage,
                arm=arm,
                label=COMPOSITE,
                context=context,
                method="weighted",
                covariates=name,
            )
            low, high = summary["a2"][context][PRIMARY_READING][arm][name]["composite_interval"]
            points[arm][name] = (w["coverage_pos"], low, high, int(w["n_effective"]))
    return points


def composite_table(summary: dict[str, Any], coverage: pd.DataFrame, context: str) -> str:
    """A2 on the composite: the single threshold and each weighting, at the smaller size."""
    rows = []
    for arm, points in composite_points(summary, coverage, context).items():
        value, low, high, _ = points["plain"]
        row = [ARM_NAMES[arm], f"{pct(value)} ({pct(low)}–{pct(high)})"]
        for name in COVARIATE_NAMES:
            value, low, high, n = points[name]
            row.append(f"{pct(value)} ({pct(low)}–{pct(high)}, n={n})")
        rows.append(row)
    header = ["Arm", "Inpatient threshold", *(f"Weighted on {v}" for v in COVARIATE_NAMES.values())]
    return _table(header, rows)


def anomaly_table(summary: dict[str, Any], coverage: pd.DataFrame, context: str) -> str:
    """A2 per judged finding, weighted on the embedding, at the smaller size.

    An asterisk marks a finding whose Holm-corrected test rejects a sensitivity of 80%.
    """
    judged = summary["judged_labels"][context]
    rows = []
    for arm in summary["arms"]:
        verdict = summary["a2"][context][PRIMARY_READING][arm][summary["verdict_covariates"]]
        row = [ARM_NAMES[arm]]
        for label in judged:
            w = _cell(
                coverage,
                arm=arm,
                label=label,
                context=context,
                method="weighted",
                covariates=summary["verdict_covariates"],
            )
            mark = "*" if verdict["anomaly_pass"][label] else ""
            row.append(f"{pct(w['coverage_pos'])}{mark} (n={int(w['n_effective'])})")
        row.append("met" if verdict["repair_holds"] else "not met")
        rows.append(row)
    header = ["Arm", *(LABEL_NAMES[k] for k in judged), "Criterion"]
    return _table(header, rows)


def readings_table(summary: dict[str, Any]) -> str:
    """The composite interval and the A2 verdict on the embedding, per size reading."""
    columns = [(c, r) for c in summary["verdict_contexts"] for r in summary["readings"]]
    rows = []
    for arm in summary["arms"]:
        row = [ARM_NAMES[arm]]
        for context, reading in columns:
            v = summary["a2"][context][reading][arm][summary["verdict_covariates"]]
            low, high = v["composite_interval"]
            mark = "met" if v["repair_holds"] else "not met"
            row.append(
                f"{pct(low)}–{pct(high)} (n={v['n']}), "
                f"{v['anomalies_passed']} of {v['anomalies_judged']} findings, {mark}"
            )
        rows.append(row)
    names = {"min_cases_kish": "smaller of cases and Kish size", "target_cases": "target cases"}
    header = ["Arm", *(f"{c.capitalize()}, {names[r]}" for c, r in columns)]
    return _table(header, rows)


def power_table(summary: dict[str, Any]) -> str:
    """Observed sensitivities that meet the composite condition, and its probability at 90%."""
    names = {"min_cases_kish": "smaller of cases and Kish size", "target_cases": "target cases"}
    rows = [
        [
            context,
            names[reading],
            str(p["n"]),
            f"{pct(p['passing_low'])}–{pct(p['passing_high'])}",
            f"{p['probability']:.2f}",
        ]
        for context, readings in summary["power"].items()
        for reading, p in readings.items()
    ]
    header = ["Target", "Reading", "n", "Sensitivity meeting the condition", "Probability"]
    return _table(header, rows)


def flags_table(summary: dict[str, Any], coverage: pd.DataFrame) -> str:
    """Sensitivity, specificity, share flagged and PPV of the composite flag.

    Sensitivity is shown without its interval, which the composite tables give at the
    size the verdict uses.
    """

    def fig(row: pd.Series, key: str) -> str:
        return f"{pct(row[key])} ({pct(row[key + '_low'])}–{pct(row[key + '_high'])})"

    rows = []
    for context in summary["verdict_contexts"]:
        for arm in summary["arms"]:
            for method, label in (("plain", "inpatient"), ("weighted", "weighted on embedding")):
                where = (
                    {"covariates": summary["verdict_covariates"]} if method == "weighted" else {}
                )
                c = _cell(
                    coverage, arm=arm, label=COMPOSITE, context=context, method=method, **where
                )
                rows.append(
                    [
                        ARM_NAMES[arm],
                        context,
                        label,
                        pct(c["coverage_pos"]),
                        fig(c, "specificity"),
                        pct(c["flagged"]),
                        fig(c, "ppv"),
                    ]
                )
    header = ["Arm", "Target", "Threshold", "Sensitivity", "Specificity", "Flagged", "PPV"]
    return _table(header, rows)


def flag_summary_table(summary: dict[str, Any], coverage: pd.DataFrame) -> str:
    """The composite flag per target and threshold: lowest and highest value across arms."""
    rows = []
    for context in summary["verdict_contexts"]:
        for method, label in (("plain", "inpatient"), ("weighted", "weighted on embedding")):
            where = {"covariates": summary["verdict_covariates"]} if method == "weighted" else {}
            cells = pd.DataFrame(
                [
                    _cell(
                        coverage, arm=arm, label=COMPOSITE, context=context, method=method, **where
                    )
                    for arm in summary["arms"]
                ]
            )
            rows.append(
                [
                    context,
                    label,
                    *(
                        f"{pct(cells[key].min())}–{pct(cells[key].max())}"
                        for key in ("coverage_pos", "specificity", "flagged", "ppv")
                    ),
                ]
            )
    header = ["Target", "Threshold", "Sensitivity", "Specificity", "Flagged", "PPV"]
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
        "composite": composite_table(summary, coverage, CONTEXT),
        "anomalies": anomaly_table(summary, coverage, CONTEXT),
        "composite_emergency": composite_table(summary, coverage, EMERGENCY),
        "anomalies_emergency": anomaly_table(summary, coverage, EMERGENCY),
        "readings": readings_table(summary),
        "power": power_table(summary),
        "flags": flags_table(summary, coverage),
        "flag_summary": flag_summary_table(summary, coverage),
        "ladder": ladder_table(summary, coverage),
        "calibration": calibration_table(summary, calibration),
        "a1": a1_table(summary),
    }

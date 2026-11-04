"""STRUCTURE.md and its supplement against the committed results of study P-A.

EchoNext cannot be committed, so nothing here recomputes a score.  What is held:
every table of the two documents is what ``ecs.structure_report`` renders from
the result files, printed once, every one-decimal percentage in either document
is printed by one of those tables, the coverage CSV holds the protocol's whole grid, and the
A2 verdicts of ``structure.json`` follow from the CSV's rows.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
import pytest
from structure_power import power

from ecs.config import REPO_ROOT
from ecs.echonext import COMPOSITE, LABELS
from ecs.structure import READINGS, a2_verdict
from ecs.structure_report import load, tables

DOC = REPO_ROOT / "STRUCTURE.md"
SUPPLEMENT = REPO_ROOT / "STRUCTURE_SUPPLEMENT.md"
CONTEXTS = ("inpatient", "emergency", "outpatient")
COVARIATES = ("age_sex", "tabular", "embedding")
PERCENT = re.compile(r"\d+\.\d%")


@pytest.fixture(scope="module")
def results() -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    return load()


@pytest.fixture(scope="module")
def rendered(results: tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]) -> dict[str, str]:
    return tables(*results)


def test_each_table_is_rendered_once_from_the_results(rendered: dict[str, str]) -> None:
    """A table edited by hand, or left behind by a re-run, would print a figure no file holds."""
    text = DOC.read_text() + SUPPLEMENT.read_text()
    printed = {name: text.count(table) for name, table in rendered.items()}
    assert printed == dict.fromkeys(rendered, 1)


def test_every_percentage_of_the_prose_is_in_a_table(rendered: dict[str, str]) -> None:
    """A figure quoted in a sentence must be one the tables print, read from the results."""
    printed = set(PERCENT.findall("\n".join(rendered.values())))
    quoted = set(PERCENT.findall(DOC.read_text() + SUPPLEMENT.read_text()))
    assert len(quoted) > 10
    assert sorted(quoted - printed) == []


def test_the_csv_holds_every_cell_of_the_grid(
    results: tuple[dict[str, Any], pd.DataFrame, dict[str, Any]],
) -> None:
    """Each arm, label and context carries plain, per-label, three weightings and five rungs."""
    summary, coverage, _ = results
    counts = coverage.groupby(["arm", "label", "context"]).size()
    assert len(counts) == len(summary["arms"]) * len(LABELS) * len(CONTEXTS)
    assert set(counts) == {2 + len(COVARIATES) + 5}


def test_the_verdicts_follow_from_the_csv(
    results: tuple[dict[str, Any], pd.DataFrame, dict[str, Any]],
) -> None:
    """Each A2 decision written to structure.json, per target and size reading, is the one
    the CSV's rows imply."""
    summary, coverage, _ = results
    assert summary["verdict_contexts"] == ["outpatient", "emergency"]
    assert summary["readings"] == READINGS
    for context in summary["verdict_contexts"]:
        judged = summary["judged_labels"][context]
        weighted = coverage[(coverage["method"] == "weighted") & (coverage["context"] == context)]
        for reading, size in READINGS.items():
            for arm in summary["arms"]:
                for name in COVARIATES:
                    rows = weighted[(weighted["arm"] == arm) & (weighted["covariates"] == name)]
                    by_label = {str(r["label"]): dict(r) for _, r in rows.iterrows()}
                    expected = a2_verdict(
                        by_label[COMPOSITE], {k: by_label[k] for k in judged}, size
                    )
                    written = summary["a2"][context][reading][arm][name]
                    assert written["repair_holds"] == expected["repair_holds"]
                    assert written["composite_pass"] == expected["composite_pass"]
                    assert written["composite_interval"] == pytest.approx(
                        expected["composite_interval"]
                    )
                    assert written["anomaly_pass"] == expected["anomaly_pass"]


def test_each_flag_agrees_with_the_coverage_of_its_row(
    results: tuple[dict[str, Any], pd.DataFrame, dict[str, Any]],
) -> None:
    """Share flagged = cases flagged + non-cases flagged, and PPV = cases flagged / flagged:
    a specificity or PPV computed on another flag than the sensitivity would break both."""
    _, coverage, _ = results
    rows = coverage[coverage["method"].isin(["plain", "perlabel", "weighted"])]
    assert rows["specificity"].notna().all()
    n_neg = rows["n"] - rows["n_pos"]
    # Pulmonary regurgitation has no outpatient case: no sensitivity, no case flagged.
    cases_flagged = (rows["coverage_pos"] * rows["n_pos"]).where(rows["n_pos"] > 0, 0.0)
    flagged = cases_flagged + (1 - rows["specificity"]) * n_neg
    assert (flagged / rows["n"]).to_numpy() == pytest.approx(rows["flagged"].to_numpy())
    has_flag = rows["flagged"] > 0
    assert (cases_flagged / flagged)[has_flag].to_numpy() == pytest.approx(
        rows.loc[has_flag, "ppv"].to_numpy()
    )


def test_the_judged_anomalies_are_those_with_fifty_target_cases(
    results: tuple[dict[str, Any], pd.DataFrame, dict[str, Any]],
) -> None:
    """The protocol judges an anomaly at 50 test cases or more in its target, fixed in advance."""
    summary, coverage, _ = results
    for context in summary["verdict_contexts"]:
        plain = coverage[
            (coverage["method"] == "plain")
            & (coverage["context"] == context)
            & (coverage["arm"] == summary["arms"][0])
        ].set_index("label")
        judged = [k for k in LABELS[:-1] if plain.loc[k, "n_pos"] >= 50]
        assert summary["judged_labels"][context] == judged
        assert len(judged) >= 1


def test_the_power_file_follows_from_structure_json(
    results: tuple[dict[str, Any], pd.DataFrame, dict[str, Any]],
) -> None:
    """Each probability of the power table is recomputed from the sizes structure.json judges."""
    summary, _, _ = results
    recorded = summary["power"]
    expected = power({k: v for k, v in summary.items() if k != "power"})
    assert recorded.keys() == expected.keys()
    for context, readings in expected.items():
        for reading, cell in readings.items():
            written = recorded[context][reading]
            assert written["n"] == cell["n"]
            assert written["passing_cases"] == cell["passing_cases"]
            assert written["probability"] == pytest.approx(cell["probability"], abs=1e-12)


def test_every_probability_of_the_prose_is_in_the_power_table(rendered: dict[str, str]) -> None:
    """« with probability 0.40 »: a power figure is read from Table 2, never typed."""
    quoted = set(re.findall(r"probability (?:above )?(?:is |of )?(0\.\d\d)", DOC.read_text()))
    assert quoted, "the text quotes no probability"
    missing = [p for p in quoted if f"| {p} |" not in rendered["power"]]
    assert not missing, f"probabilities in the prose but not in the power table: {missing}"

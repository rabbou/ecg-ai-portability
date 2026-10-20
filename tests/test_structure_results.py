"""STRUCTURE.md against the committed results of study P-A.

EchoNext cannot be committed, so nothing here recomputes a score.  What is held:
every table of the document is what ``ecs.structure_report`` renders from the
three result files, every one-decimal percentage in the document is printed by
one of those tables, the coverage CSV holds the protocol's whole grid, and the
A2 verdicts of ``structure.json`` follow from the CSV's rows.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
import pytest

from ecs.config import REPO_ROOT
from ecs.echonext import COMPOSITE, LABELS
from ecs.structure import a2_verdict
from ecs.structure_report import load, tables

DOC = REPO_ROOT / "STRUCTURE.md"
CONTEXTS = ("inpatient", "emergency", "outpatient")
COVARIATES = ("age_sex", "tabular", "embedding")
PERCENT = re.compile(r"\d+\.\d%")


@pytest.fixture(scope="module")
def results() -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    return load()


@pytest.fixture(scope="module")
def rendered(results: tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]) -> dict[str, str]:
    return tables(*results)


def test_each_table_of_the_document_is_rendered_from_the_results(
    rendered: dict[str, str],
) -> None:
    """A table edited by hand, or left behind by a re-run, would print a figure no file holds."""
    text = DOC.read_text()
    missing = [name for name, table in rendered.items() if table not in text]
    assert missing == []


def test_every_percentage_of_the_prose_is_in_a_table(rendered: dict[str, str]) -> None:
    """A figure quoted in a sentence must be one the tables print, read from the results."""
    printed = set(PERCENT.findall("\n".join(rendered.values())))
    quoted = set(PERCENT.findall(DOC.read_text()))
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
    """The A2 decision written to structure.json is the one the CSV's rows imply."""
    summary, coverage, _ = results
    weighted = coverage[(coverage["method"] == "weighted") & (coverage["context"] == "outpatient")]
    for arm in summary["arms"]:
        for name in COVARIATES:
            rows = weighted[(weighted["arm"] == arm) & (weighted["covariates"] == name)]
            by_label = {str(r["label"]): dict(r) for _, r in rows.iterrows()}
            expected = a2_verdict(
                by_label[COMPOSITE], {k: by_label[k] for k in summary["judged_labels"]}
            )
            assert summary["a2"][arm][name]["repair_holds"] == expected["repair_holds"]
            assert summary["a2"][arm][name]["composite_pass"] == expected["composite_pass"]
            assert summary["a2"][arm][name]["composite_interval"] == pytest.approx(
                expected["composite_interval"]
            )
            assert summary["a2"][arm][name]["anomaly_pass"] == expected["anomaly_pass"]


def test_the_judged_anomalies_are_those_with_fifty_outpatient_cases(
    results: tuple[dict[str, Any], pd.DataFrame, dict[str, Any]],
) -> None:
    """The protocol judges an anomaly at 50 test outpatient cases or more, fixed in advance."""
    summary, coverage, _ = results
    plain = coverage[
        (coverage["method"] == "plain")
        & (coverage["context"] == "outpatient")
        & (coverage["arm"] == summary["arms"][0])
    ].set_index("label")
    judged = [k for k in LABELS[:-1] if plain.loc[k, "n_pos"] >= 50]
    assert summary["judged_labels"] == judged
    assert len(judged) >= 1

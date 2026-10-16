"""The committed MIMIC-IV-ECG label table, and its rebuild from the release.

The table tests read the committed ``results/mimic_label_map.json``; rebuild it
with

    .venv/bin/python scripts/mimic_label_table.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from ecs.config import MIMIC_ECG_DIR, RESULTS_DIR
from ecs.mimic import (
    AMBIGUITIES,
    CLASSES,
    POSITIVE_STATUSES,
    classify,
    record_labels,
    record_statuses,
)

PATH = Path(RESULTS_DIR) / "mimic_label_map.json"


@pytest.fixture(scope="module")
def table() -> dict[str, Any]:
    if not PATH.exists():
        pytest.skip(f"{PATH} is not built; run scripts/mimic_label_table.py")
    return dict(json.loads(PATH.read_text()))


# The counts the committed table holds, pinned so a change of rule that moves
# them has to change this line too.
PINNED_RECORDS = {
    "NSR": 461068,
    "AF": 78069,
    "LBBB": 23451,
    "RBBB": 57360,
    "IAVB": 66689,
    "MI_OLD": 70388,
}


class TestTheCommittedTable:
    def test_it_covers_the_whole_release_read_from_digest_checked_files(
        self, table: dict[str, Any]
    ) -> None:
        assert table["n_records"] == 800035 and table["n_patients"] == 161352
        for name, entry in table["files"].items():
            assert entry["matches_release_sha256sums"], name

    def test_the_count_per_diagnosis_is_the_pinned_one(self, table: dict[str, Any]) -> None:
        counts = {k: table["classes"][k]["n_records"] for k in CLASSES}
        assert counts == PINNED_RECORDS

    def test_the_strict_count_never_exceeds_the_positive_one(self, table: dict[str, Any]) -> None:
        for klass in CLASSES:
            block = table["classes"][klass]
            assert block["strict_n_records"] <= block["n_records"]
            assert block["n_patients"] <= block["n_records"]
            if POSITIVE_STATUSES[klass] == {"definite"}:
                assert block["strict_n_records"] == block["n_records"]

    def test_every_ambiguous_join_is_listed_with_its_count(self, table: dict[str, Any]) -> None:
        listed = {a["key"]: a for a in table["ambiguities"]}
        assert set(listed) == {a.key for a in AMBIGUITIES}
        for key, entry in listed.items():
            assert entry["n_records"] > 0, key
            assert entry["decision"] and entry["why"], key

    def test_the_statement_table_is_what_the_rules_give_today(self, table: dict[str, Any]) -> None:
        for row in table["statements"]:
            assert classify(row["statement"]) == row["status"], row["statement"]
        assert len(table["statements"]) == table["n_distinct_statements_touching_a_class"]

    def test_borderline_first_degree_block_has_a_pr_above_200_ms(
        self, table: dict[str, Any]
    ) -> None:
        """The reason borderline counts as IAVB, held to the cart's own measurement."""
        entry = next(a for a in table["ambiguities"] if a["key"] == "iavb-borderline")
        measured = entry["measured_intervals_ms"]
        assert measured["interval"] == "pr"
        assert measured["median"] == 208.0
        assert measured["share_above_200"] >= 0.975

    def test_a_bundle_branch_conduction_defect_has_a_wide_qrs(self, table: dict[str, Any]) -> None:
        entry = next(a for a in table["ambiguities"] if a["key"] == "bbb-type-conduction-defect")
        measured = entry["measured_intervals_ms"]
        assert measured["interval"] == "qrs" and measured["median"] == 140.0
        assert measured["share_at_least_120"] >= 0.96


@pytest.mark.data
class TestTheTableRebuildsFromTheRelease:
    def test_the_counts_come_back_from_the_csv(self, table: dict[str, Any]) -> None:
        source = Path(MIMIC_ECG_DIR) / "machine_measurements.csv"
        if not source.exists():
            pytest.skip(f"{source} is not on disk")
        frame = pd.read_csv(source, low_memory=False)
        labels = record_labels(record_statuses(frame))
        assert {k: int(labels[k].sum()) for k in CLASSES} == PINNED_RECORDS

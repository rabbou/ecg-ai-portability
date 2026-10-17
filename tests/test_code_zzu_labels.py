"""The committed CODE and ZZU label table against what each dataset's own documentation says.

The expected counts are typed here from the documents, not read from the
table's ``documented`` block, so a table and a block that drifted together
would still fail:

- CODE test set: Ribeiro et al., Nat Commun 11, 1760 (2020), Table 1, test column.
- ZZU pECG: Tan et al., Sci Data 2025, doi:10.1038/s41597-025-05225-z,
  Supplementary Table 1, statement counts over all 14,190 records.
- CODE-15%: no table of its own; Ribeiro et al. 2020, Table 1, gives the
  prevalences of the full CODE dataset it was sampled from, to one decimal of a
  percent, so the check is to within 0.3 points and is coarse by construction.

Rebuild the table with ``scripts/code_zzu_label_table.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from ecs.code_ecg import code_labels, code_test_frame
from ecs.config import CODE15_DIR, CODE_TEST_DIR, RESULTS_DIR, ZZU_DIR
from ecs.zzu import zzu_labels

PATH = Path(RESULTS_DIR) / "code_zzu_label_map.json"


@pytest.fixture(scope="module")
def table() -> dict[str, Any]:
    return json.loads(PATH.read_text())


class TestTheDocumentedCounts:
    def test_the_code_test_set_holds_the_published_counts(self, table: dict[str, Any]) -> None:
        """A label read off the wrong column would change the cardiologist reference."""
        counts = table["corpora"]["code_test"]["release"]["counts"]
        assert counts == {"AF": 13, "LBBB": 30, "RBBB": 34, "IAVB": 28}
        assert table["corpora"]["code_test"]["release"]["n_exams"] == 827

    def test_zzu_holds_the_statement_counts_of_its_supplement(self, table: dict[str, Any]) -> None:
        """A mapping that missed a modifier or the uncoded first-degree block undercounts."""
        statements = table["corpora"]["zzu"]["release"]["statements"]
        assert statements == {
            "E50": 31,
            "I104": 31,
            "I106": 395,
            "First-degree AV block (H74)": 165,
            "H82": 12,
        }
        records = table["corpora"]["zzu"]["release"]["record_counts"]
        assert (records["AF"], records["LBBB"], records["RBBB"]) == (31, 31, 395)
        assert table["corpora"]["zzu"]["release"]["n_records"] == 14_190

    def test_code15_prevalences_are_those_of_the_code_it_was_drawn_from(
        self, table: dict[str, Any]
    ) -> None:
        """A flag read inverted or off the wrong column would be far from CODE's prevalence."""
        release = table["corpora"]["code15"]["release"]
        assert (release["n_exams"], release["n_patients"]) == (345_779, 233_770)
        full_code = {"IAVB": 0.015, "RBBB": 0.027, "LBBB": 0.017, "AF": 0.018}
        for key, published in full_code.items():
            assert abs(release["prevalence"][key] - published) < 0.003, key


class TestTheSample:
    def test_the_sample_is_the_stated_size_one_exam_per_patient(
        self, table: dict[str, Any]
    ) -> None:
        """The document states 22,000 patients and seed 27; the table must say the same."""
        sample = table["corpora"]["code15"]["sample"]
        assert (sample["n"], sample["seed"]) == (22_000, 27)
        assert sum(sample["drawn_per_stratum"].values()) == 22_000

    def test_the_sample_keeps_the_pool_prevalence_to_one_patient_per_stratum(
        self, table: dict[str, Any]
    ) -> None:
        """A stratified draw that lost its strata would drift from the pool it is drawn from."""
        code15 = table["corpora"]["code15"]
        sample, pool = code15["sample"], code15["sample"]["pool"]
        # Largest-remainder rounding moves each stratum by under one patient, and 16
        # strata at most, so a flag's count moves by under 16 of 22,000.
        for key in ("AF", "LBBB", "RBBB", "IAVB"):
            assert abs(sample["prevalence"][key] - pool["prevalence"][key]) < 16 / 22_000, key

    def test_nsr_is_refused_on_all_three_corpora(self, table: dict[str, Any]) -> None:
        """Filling NSR from 'normal ECG' would put a narrower class on these corpora."""
        assert sorted(map(tuple, table["refused"])) == [
            ("code15", "NSR"),
            ("code_test", "NSR"),
            ("zzu", "NSR"),
        ]


@pytest.mark.data
class TestTheTableRebuildsFromTheReleases:
    def test_the_code_test_counts_rebuild(self, table: dict[str, Any]) -> None:
        """The committed counts must be the release's, read through the module."""
        _, labels = code_test_frame(CODE_TEST_DIR)
        counts = {k: int(labels[k].sum()) for k in ("AF", "LBBB", "RBBB", "IAVB")}
        assert counts == table["corpora"]["code_test"]["release"]["counts"]

    def test_the_code15_and_zzu_counts_rebuild(self, table: dict[str, Any]) -> None:
        """The committed CODE-15% and ZZU counts must be the releases'."""
        exams = pd.read_csv(CODE15_DIR / "exams.csv")
        labels = code_labels(exams)
        counts = {k: int(labels[k].sum()) for k in ("AF", "LBBB", "RBBB", "IAVB")}
        assert counts == table["corpora"]["code15"]["release"]["counts"]
        zzu = zzu_labels(pd.read_csv(ZZU_DIR / "AttributesDictionary.csv")["AHA_code"])
        counts = {k: int(zzu[k].sum()) for k in ("AF", "LBBB", "RBBB", "IAVB")}
        assert counts == table["corpora"]["zzu"]["release"]["record_counts"]

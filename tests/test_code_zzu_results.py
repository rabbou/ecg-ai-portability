"""Every figure of CODE_ZZU.md, recomputed from the committed scores of results/rotation_seven.

The coverage tables, the counts of cells below 90%, the AUROC medians and the
CODE-15% home figures are rebuilt with ``ecs.transfer`` from the score arrays,
never read from the summary files, and the document must print each one.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from code_zzu_rotation import ROTATION_SEVEN, below, target_arrays
from mimic_rotation import load
from sklearn.metrics import roc_auc_score

from ecs.config import RESULTS_DIR
from ecs.provenance import REPO_ROOT
from ecs.rotation import EXTENDED_SOURCES, class_keys, usable_classes
from ecs.transfer import ALPHA, conformal_sets, coverage_row

DOC = (REPO_ROOT / "CODE_ZZU.md").read_text()
SOURCES = {
    "PTB-XL": "ptbxl",
    "Shandong": "sph",
    "Chapman-Shaoxing and Ningbo": "chapman_ningbo",
    "Georgia": "georgia",
    "CPSC": "cpsc",
    "Beth Israel": "mimic",
    "CODE-15%": "code15",
}
LABELS = ("AF", "LBBB", "RBBB", "IAVB")


TARGETS = ("code15", "code_test", "code_test_10s", "zzu")


def perlabel_rows(source: str, target: str) -> list[dict[str, Any]]:
    """The per-label rule's coverage with Wilson intervals, and the point AUROC.

    Rebuilt from the score arrays with ``ecs.transfer`` like the published table,
    without its bootstrap of the AUROC interval, which no figure of the document
    prints and which would cost minutes per run.
    """
    y_cal, p_cal = load(ROTATION_SEVEN / source / "scores" / f"{source}_cal.npz")
    y, p = target_arrays(ROTATION_SEVEN, source, target)
    corpus = "code_test" if target == "code_test_10s" else target
    out = []
    for key in [k for k in usable_classes(source) if k in usable_classes(corpus)]:
        j = class_keys().index(key)
        sets = conformal_sets(p_cal[:, j], y_cal[:, j], p[:, j], ALPHA)["perlabel"]
        row = coverage_row(sets, y[:, j])
        auroc = roc_auc_score(y[:, j], p[:, j]) if 0 < y[:, j].sum() < len(y) else None
        out.append(
            {
                "source": source,
                "target": target,
                "label": key,
                "method": "perlabel",
                "auroc": auroc,
                **row,
            }
        )
    return out


@pytest.fixture(scope="module")
def rows() -> list[dict[str, Any]]:
    if not (ROTATION_SEVEN / "code15" / "scores" / "code15_cal.npz").exists():
        pytest.skip("results/rotation_seven is not committed in this tree")
    into_new = [r for s in SOURCES.values() for t in TARGETS for r in perlabel_rows(s, t)]
    out_of_code15 = [r for t in EXTENDED_SOURCES for r in perlabel_rows("code15", t)]
    return into_new + out_of_code15


def cell(rows: list[dict[str, Any]], source: str, target: str, label: str) -> dict[str, Any]:
    (row,) = [r for r in rows if (r["source"], r["target"], r["label"]) == (source, target, label)]
    return row


def table_after(caption_start: str) -> dict[str, list[str]]:
    """The markdown table that ends just before the caption starting with these words."""
    block = DOC[: DOC.index(caption_start)].rstrip().split("\n")
    lines: list[str] = []
    while block and block[-1].startswith("|"):
        lines.insert(0, block.pop())
    return {
        cells[0]: cells[1:]
        for cells in ([c.strip() for c in line.strip("|").split("|")] for line in lines[2:])
    }


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


@pytest.mark.parametrize(
    ("caption", "target"),
    [
        ("Coverage of the cases in the CODE test set", "code_test"),
        ("Coverage of the cases in ZZU pECG", "zzu"),
    ],
)
def test_each_coverage_table_comes_back_from_the_scores(
    rows: list[dict[str, Any]], caption: str, target: str
) -> None:
    """A table typed by hand, or left behind by a rerun, would publish a figure no score holds."""
    table = table_after(caption)
    assert len(table) >= 6
    for name, printed in table.items():
        expected = [pct(cell(rows, SOURCES[name], target, k)["coverage_pos"]) for k in LABELS]
        assert printed == expected, name


def test_the_counts_of_cells_below_ninety_come_back(rows: list[dict[str, Any]]) -> None:
    """The opening's 56 of 80, 46, 24 of 28 and 30 of 32 must be what the scores give."""
    adult = [r for r in rows if r["target"] in ("code15", "code_test", "code_test_10s")]
    into_new = below([r for r in adult if r["source"] != r["target"]])
    assert f"in {into_new['below_90']} of {into_new['cells']} source" in DOC
    assert f"{into_new['interval_below_90']} of them with the whole 95% interval" in DOC
    zzu = below([r for r in rows if r["target"] == "zzu"])
    assert f"miss 90% in {zzu['below_90']} of {zzu['cells']} cells" in DOC
    out = below(
        [r for r in rows if r["source"] == "code15" and r["target"] not in ("code15", "zzu")]
    )
    assert f"miss 90% in {out['below_90']} of {out['cells']} cells" in DOC


def test_the_auroc_medians_come_back(rows: list[dict[str, Any]]) -> None:
    """The discrimination the document reads on the CODE test set and ZZU."""
    foreign = [s for s in SOURCES.values() if s != "code15"]
    for target, label in (
        ("code_test", "AF"),
        ("code_test", "LBBB"),
        ("code_test", "RBBB"),
        ("code_test", "IAVB"),
        ("zzu", "RBBB"),
        ("zzu", "IAVB"),
    ):
        median = np.median([cell(rows, s, target, label)["auroc"] for s in foreign])
        assert f"{median:.3f}" in DOC, (target, label)


def test_the_code15_home_figures_come_back(rows: list[dict[str, Any]]) -> None:
    """The home coverage range and the share sent to a reader under a model that did not learn."""
    home = [cell(rows, "code15", "code15", k) for k in LABELS]
    low, high = min(r["coverage_pos"] for r in home), max(r["coverage_pos"] for r in home)
    assert f"cover {pct(low)} to {pct(high)} of the cases" in DOC
    sent = [r["abstention"] for r in home]
    assert f"sending {round(100 * min(sent))}% to {round(100 * max(sent))}% of tracings" in DOC
    metrics = json.loads((ROTATION_SEVEN / "code15" / "metrics.json").read_text())
    assert f"macro AUROC of {metrics['val_macro_auroc']:.3f}" in DOC


def test_the_label_counts_printed_are_the_table_s() -> None:
    """The document's label table must be the committed label map's."""
    table = json.loads((Path(RESULTS_DIR) / "code_zzu_label_map.json").read_text())["corpora"]
    test = table["code_test"]["release"]["counts"]
    printed = table_after("The counts reproduce")["CODE test set, 827 exams"][:4]
    assert printed == [str(test[k]) for k in LABELS]
    shares = table["code15"]["release"]["prevalence"]
    printed = table_after("The counts reproduce")["CODE-15%, share of 345,779 exams"][:4]
    assert printed == [f"{100 * shares[k]:.2f}%" for k in LABELS]
    assert re.search(r"Only 132,161 exams hold ten seconds", DOC)
    assert table["code15"]["spans"]["n_ten_seconds"] == 132_161

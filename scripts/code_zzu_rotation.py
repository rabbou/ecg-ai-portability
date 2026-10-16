"""The rotation with CODE-15% as a seventh source, the CODE test set and ZZU as targets.

The six sources of ``results/rotation_six`` and CODE-15% are trained again
under ``results/rotation_seven`` with the same recipe, because the six-corpus
checkpoints were not kept.  Every source's thresholds are fitted once, on its
whole calibration part, and spent unchanged on every test part: the seven
sources' own, the CODE test set's 827 exams and ZZU's children.  Per pair,
diagnosis and rule (``plain``, ``pooled``, ``perlabel``, at the 90% level), the
coverage of the tracings with and without the diagnosis, each with its 95%
Wilson interval, as in ``scripts/mimic_rotation.py``.

The CODE test set is also read on its ten-second exams alone (target
``code_test_10s``), because 539 of its exams were zero-filled to ten seconds.
ZZU's rows are a side arm and enter no adult summary.

The same estimator is run on the ``rotation_six`` scores, and the difference on
the cells the two runs share is how far a third training of the same recipe
moves a figure.

Writes ``results/code_zzu_rotation.csv`` and ``results/code_zzu_rotation.json``.

Usage: .venv/bin/python scripts/code_zzu_rotation.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from mimic_rotation import COLUMNS, NAMES, load, rounded
from numpy.typing import NDArray

from ecs.config import RESULTS_DIR
from ecs.encoders import SAW
from ecs.provenance import provenance_block
from ecs.rotation import (
    CODE_ZZU_SOURCES,
    EXTENDED_SOURCES,
    SIDE_ARM,
    TARGET_ONLY,
    class_keys,
    usable_classes,
)
from ecs.transfer import ALPHA, METHODS, auroc_row, conformal_sets, coverage_row

ROTATION_SEVEN = Path(RESULTS_DIR) / "rotation_seven"
ROTATION_SIX = Path(RESULTS_DIR) / "rotation_six"
LABEL_MAP = Path(RESULTS_DIR) / "code_zzu_label_map.json"
OUT_CSV = Path(RESULTS_DIR) / "code_zzu_rotation.csv"
OUT_JSON = Path(RESULTS_DIR) / "code_zzu_rotation.json"
HEADLINE = "perlabel"
NEW = ("code15", "code_test", "code_test_10s", "zzu")
TARGETS = (*CODE_ZZU_SOURCES, *TARGET_ONLY)

PRODUCERS = [
    "scripts/code_zzu_rotation.py",
    "scripts/mimic_rotation.py",
    "src/ecs/code_ecg.py",
    "src/ecs/conformal.py",
    "src/ecs/encoders.py",
    "src/ecs/metrics.py",
    "src/ecs/rotation.py",
    "src/ecs/small_set.py",
    "src/ecs/transfer.py",
    "src/ecs/zzu.py",
]

CORPUS_NAMES = {
    **NAMES,
    "code15": "CODE-15% (Telehealth Network of Minas Gerais)",
    "code_test": "CODE test set (Telehealth Network of Minas Gerais, 2018)",
    "code_test_10s": "CODE test set, ten-second exams only",
    "zzu": "ZZU pECG (First Affiliated Hospital of Zhengzhou University, children)",
}


def ten_second_mask(rotation: Path, source: str) -> NDArray[np.bool_]:
    """Which rows of a CODE test score file are ten-second exams, by id."""
    short = set(json.loads(LABEL_MAP.read_text())["corpora"]["code_test"]["short_exam_ids"])
    with np.load(rotation / source / "scores" / "code_test_test.npz") as data:
        return np.array([str(i) not in short for i in data["ids"]])


def target_arrays(
    rotation: Path, source: str, target: str
) -> tuple[NDArray[np.int_], NDArray[np.float64]]:
    if target == "code_test_10s":
        y, p = load(rotation / source / "scores" / "code_test_test.npz")
        keep = ten_second_mask(rotation, source)
        return y[keep], p[keep]
    return load(rotation / source / "scores" / f"{target}_test.npz")


def pair_rows(rotation: Path, source: str, target: str) -> list[dict[str, Any]]:
    """Every label and rule for one source spent on one target."""
    y_cal, p_cal = load(rotation / source / "scores" / f"{source}_cal.npz")
    y_tgt, p_tgt = target_arrays(rotation, source, target)
    corpus = "code_test" if target == "code_test_10s" else target
    shared = [k for k in usable_classes(source) if k in usable_classes(corpus)]
    rows = []
    for key in shared:
        j = class_keys().index(key)
        sets = conformal_sets(p_cal[:, j], y_cal[:, j], p_tgt[:, j], ALPHA)
        auc = auroc_row(y_tgt[:, j], p_tgt[:, j])
        for method in METHODS:
            row = coverage_row(sets[method], y_tgt[:, j])
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "role": "home" if source == target else "away",
                    "label": key,
                    "method": method,
                    "n": row["n"],
                    "n_pos": row["n_pos"],
                    "prevalence": round(row["n_pos"] / row["n"], 5) if row["n"] else None,
                    "auroc": auc["auroc"],
                    "coverage_pos": row["coverage_pos"],
                    "coverage_pos_low": row["coverage_pos_low"],
                    "coverage_pos_high": row["coverage_pos_high"],
                    "coverage_neg": row["coverage_neg"],
                    "coverage_neg_low": row["coverage_neg_low"],
                    "coverage_neg_high": row["coverage_neg_high"],
                    "abstention": row["abstention"],
                    "n_cal_pos": int(y_cal[:, j].sum()),
                }
            )
    return rows


def all_rows(rotation: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in CODE_ZZU_SOURCES:
        for target in (*TARGETS[: TARGETS.index("code_test") + 1], "code_test_10s", *SIDE_ARM):
            rows.extend(pair_rows(rotation, source, target))
    return rows


def below(rows: list[dict[str, Any]]) -> dict[str, int]:
    """How many headline cells miss 90%, and how many with their whole interval below."""
    cells = [r for r in rows if r["method"] == HEADLINE and r["n_pos"]]
    return {
        "cells": len(cells),
        "below_90": sum(r["coverage_pos"] < 1 - ALPHA for r in cells),
        "interval_below_90": sum(r["coverage_pos_high"] < 1 - ALPHA for r in cells),
    }


def reproduction(seven: list[dict[str, Any]]) -> dict[str, Any]:
    """The six-corpus cells, trained twice with the same recipe, at the headline rule."""
    six: list[dict[str, Any]] = []
    for source in EXTENDED_SOURCES:
        for target in EXTENDED_SOURCES:
            six.extend(
                r for r in pair_rows(ROTATION_SIX, source, target) if r["method"] == HEADLINE
            )
    again = {(r["source"], r["target"], r["label"]): r for r in seven if r["method"] == HEADLINE}
    gaps = [
        abs(again[(r["source"], r["target"], r["label"])]["coverage_pos"] - r["coverage_pos"])
        for r in six
    ]
    return {
        "what": (
            "coverage of the diagnosis under the per-label rule, rotation_six against "
            "rotation_seven, same records, same recipe, trained twice"
        ),
        "n_cells": len(gaps),
        "median_absolute_difference": round(float(np.median(gaps)), 4),
        "p90_absolute_difference": round(float(np.quantile(gaps, 0.9)), 4),
        "max_absolute_difference": round(float(max(gaps)), 4),
    }


def main() -> int:
    rows = all_rows(ROTATION_SEVEN)
    with OUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(rounded(r) for r in rows)
    adult = [r for r in rows if r["target"] not in SIDE_ARM]
    into_new = [r for r in adult if r["target"] in NEW and r["source"] != r["target"]]
    from_code15 = [r for r in adult if r["source"] == "code15" and r["target"] != "code15"]
    configs = {
        s: json.loads((ROTATION_SEVEN / s / "config.json").read_text()) for s in CODE_ZZU_SOURCES
    }
    table: dict[str, Any] = {
        "written_by": "scripts/code_zzu_rotation.py",
        "provenance": provenance_block(PRODUCERS),
        "settings": {
            "alpha": ALPHA,
            "methods": list(METHODS),
            "headline_method": HEADLINE,
            "interval": "Wilson score interval, 95%, over the target's test tracings",
            "calibration": "the source's whole calibration part, thresholds fitted once",
            "corpus_names": CORPUS_NAMES,
            "classes": class_keys(),
            "side_arm": list(SIDE_ARM),
        },
        "label_sources": {
            "code15": "CODE's reconciled report labels (results/code_zzu_label_map.json)",
            "code_test": "two cardiologists, a third settling disagreements",
            "zzu": "ECG-room physicians' statements as AHA codes",
            "mimic": "machine statements (results/mimic_label_map.json)",
            "others": "SNOMED or AHA codes of each release (results/label_map.json)",
        },
        "encoders_that_saw_code15": sorted(a for a, seen in SAW.items() if "code15" in seen),
        "sources": {
            s: {
                "split_sizes": c["split_sizes"],
                "standardisation": c["standardisation"],
                "deviations": c["deviations"],
                "machine": c["machine"],
            }
            for s, c in configs.items()
        },
        "grid": {"file": "results/code_zzu_rotation.csv", "columns": list(COLUMNS)},
        "summary": {
            "into_new_adult_targets": below(into_new),
            "from_code15": below(from_code15),
            "into_zzu": below([r for r in rows if r["target"] in SIDE_ARM]),
        },
        "reproduction": reproduction(rows),
    }
    OUT_JSON.write_text(json.dumps(table, indent=1) + "\n")
    print(f"wrote {OUT_CSV} ({len(rows)} rows) and {OUT_JSON}")
    print(json.dumps(table["summary"], indent=1), json.dumps(table["reproduction"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

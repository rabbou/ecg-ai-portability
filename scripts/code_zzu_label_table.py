"""The label table of CODE-15%, the CODE test set and ZZU pECG, and the CODE-15% sample.

For each corpus: the count of every rotation class over the whole release, over
the records the rotation reads, and the count the dataset's own documentation
gives, with where it gives it.  For CODE-15%, the sample itself: how many exams
hold ten seconds, how many patients keep one, the strata and what each
contributed, and the digest of the chosen exam ids.

``--ids-out`` writes the chosen exam ids, one per line, which is what
``scripts/code15_extract.py extract`` reads on the machine that holds the zips.

Writes ``results/code_zzu_label_map.json``.

Usage: .venv/bin/python scripts/code_zzu_label_table.py --spans ~/data/code15-sample/spans.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from ecs import code_ecg, zzu
from ecs.code_ecg import (
    CODE_FLAGS,
    TEN_SECONDS_AT_400,
    code15_pool,
    code15_sample,
    code_labels,
    strata_of,
)
from ecs.config import CODE15_DIR, CODE_TEST_DIR, RESULTS_DIR, ZZU_DIR
from ecs.provenance import provenance_block
from ecs.rotation import corpus_index
from ecs.zzu import IAVB_STATEMENT, aha_tokens, zzu_labels

OUT = Path(RESULTS_DIR) / "code_zzu_label_map.json"
CLASSES = ("AF", "LBBB", "RBBB", "IAVB")

PRODUCERS = [
    "scripts/code_zzu_label_table.py",
    "src/ecs/code_ecg.py",
    "src/ecs/mimic.py",
    "src/ecs/rotation.py",
    "src/ecs/small_set.py",
    "src/ecs/zzu.py",
]

# What each dataset's own documentation says, typed from the page named beside it.
DOCUMENTED: dict[str, dict[str, Any]] = {
    "code15": {
        "source": (
            "Ribeiro et al., Nat Commun 11, 1760 (2020), Table 1, train and validation "
            "(n = 2,322,513): the full CODE dataset CODE-15% was sampled from; CODE-15% "
            "publishes no table of its own"
        ),
        "prevalence": {"IAVB": 0.015, "RBBB": 0.027, "LBBB": 0.017, "AF": 0.018},
    },
    "code_test": {
        "source": "Ribeiro et al., Nat Commun 11, 1760 (2020), Table 1, test (n = 827)",
        "counts": {"IAVB": 28, "RBBB": 34, "LBBB": 30, "AF": 13},
    },
    "zzu": {
        "source": (
            "Tan et al., Sci Data 2025, doi:10.1038/s41597-025-05225-z, Supplementary "
            "Table 1, statement counts over all 14,190 records"
        ),
        "statements": {
            "AF": {"E50": 31},
            "LBBB": {"I104": 31},
            "RBBB": {"I106": 395},
            "IAVB": {"First-degree AV block (H74)": 165, "H82": 12},
        },
    },
}


def counts(labels: pd.DataFrame) -> dict[str, int]:
    return {key: int(labels[key].sum()) for key in CLASSES}


def prevalences(labels: pd.DataFrame) -> dict[str, float]:
    return {key: round(float(labels[key].mean()), 5) for key in CLASSES}


def rotation_counts(corpus: str) -> dict[str, Any]:
    """Counts per part of the rows the rotation reads."""
    frame = corpus_index(corpus).frame
    return {
        part: {"n": int((frame["part"] == part).sum()), **counts(frame[frame["part"] == part])}
        for part in ("train", "val", "cal", "test")
        if (frame["part"] == part).any()
    }


def code15_table(spans_path: Path, ids_out: Path | None) -> dict[str, Any]:
    exams = pd.read_csv(CODE15_DIR / "exams.csv")
    spans = pd.read_csv(spans_path).set_index("exam_id")["span"]
    release = code_labels(exams)
    ten = spans[spans >= TEN_SECONDS_AT_400]
    sample = code15_sample(exams, spans)
    ids = "\n".join(str(int(e)) for e in sample.index) + "\n"
    if ids_out is not None:
        ids_out.write_text(ids)
    eligible = exams[exams["exam_id"].isin(ten.index)]
    pool = code15_pool(exams, spans)
    sample_labels = code_labels(sample)
    return {
        "release": {
            "n_exams": len(exams),
            "n_patients": int(exams["patient_id"].nunique()),
            "counts": counts(release),
            "prevalence": prevalences(release),
        },
        "spans": {
            "file_sha256": hashlib.sha256(spans_path.read_bytes()).hexdigest(),
            "n_exams": len(spans),
            "n_ten_seconds": len(ten),
            "span_counts": {str(k): int(v) for k, v in spans.value_counts().head(6).items()},
            "n_patients_with_a_ten_second_exam": int(eligible["patient_id"].nunique()),
        },
        "sample": {
            "n": len(sample),
            "seed": code_ecg.CODE15_SAMPLE_SEED,
            "strata": "the four flags AF, LBBB, RBBB, 1dAVb of each patient's drawn exam",
            "strata_order": list(CODE_FLAGS),
            "pool": {
                "what": "one ten-second exam per patient who has one, the frame the draw is from",
                "n": len(pool),
                "per_stratum": {str(k): int(v) for k, v in strata_of(pool).value_counts().items()},
                "prevalence": prevalences(code_labels(pool)),
            },
            "drawn_per_stratum": {
                str(k): int(v) for k, v in strata_of(sample).value_counts().items()
            },
            "ids_sha256": hashlib.sha256(ids.encode()).hexdigest(),
            "counts": counts(sample_labels),
            "prevalence": prevalences(sample_labels),
        },
    }


def code_test_table() -> dict[str, Any]:
    frame, labels = code_ecg.code_test_frame(CODE_TEST_DIR)
    short = pd.Series(code_ecg.short_exams(list(frame["path"])), index=frame.index)
    return {
        "release": {"n_exams": len(labels), "counts": counts(labels)},
        "short_exams": int(short.sum()),
        "short_exam_ids": [str(i) for i in frame.index[short.to_numpy()]],
        "ten_second_exams": {"n": int((~short).sum()), **counts(labels[~short])},
    }


def zzu_table() -> dict[str, Any]:
    table = pd.read_csv(ZZU_DIR / "AttributesDictionary.csv")
    tokens = table["AHA_code"].apply(aha_tokens)
    labels = zzu_labels(table["AHA_code"])
    twelve = table["Lead"] == 12
    return {
        "release": {
            "n_records": len(table),
            "n_patients": int(table["Patient_ID"].nunique()),
            "n_twelve_lead": int(twelve.sum()),
            "record_counts": counts(labels),
            "statements": {
                "E50": int(tokens.apply(lambda t: any(x.startswith("E50") for x in t)).sum()),
                "I104": int(tokens.apply(lambda t: "I104" in t).sum()),
                "I106": int(tokens.apply(lambda t: "I106" in t).sum()),
                "First-degree AV block (H74)": int(
                    tokens.apply(lambda t: IAVB_STATEMENT in t).sum()
                ),
                "H82": int(tokens.apply(lambda t: "H82" in t).sum()),
            },
        },
        "twelve_lead_records": counts(labels[twelve]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spans", type=Path, required=True, help="what code15_extract.py spans wrote"
    )
    parser.add_argument("--ids-out", type=Path, help="write the sampled CODE-15%% exam ids here")
    parser.add_argument(
        "--no-rotation", action="store_true", help="skip the rotation rows (no sample on disk yet)"
    )
    args = parser.parse_args()
    corpora = {
        "code15": code15_table(args.spans, args.ids_out),
        "code_test": code_test_table(),
        "zzu": zzu_table(),
    }
    if not args.no_rotation:
        for corpus, table in corpora.items():
            table["rotation"] = rotation_counts(corpus)
    out = {
        "written_by": "scripts/code_zzu_label_table.py",
        "provenance": provenance_block(PRODUCERS),
        "classes": list(CLASSES),
        "refused": [list(pair) for pair in (*code_ecg.REFUSED, *zzu.REFUSED)],
        "ambiguities": [asdict(a) for a in (*code_ecg.AMBIGUITIES, *zzu.AMBIGUITIES)],
        "documented": DOCUMENTED,
        "corpora": corpora,
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({c: t.get("release", {}) for c, t in corpora.items()}, indent=1))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

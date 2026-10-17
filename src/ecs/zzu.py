"""ZZU pECG read as a side arm: children, not one more adult hospital.

The ZZU pECG release (figshare 27078763, CC BY 4.0; Tan et al., Sci Data 2025)
holds 14,190 ECGs of 11,643 children aged 0 to 14 admitted to the First
Affiliated Hospital of Zhengzhou University, 2018-2024: 12,334 twelve-lead and
1,856 nine-lead records, WFDB at 500 Hz, 5 to 120 seconds long.  Its statements
were written in Chinese by the ECG room's physicians and converted to the AHA
standard's codes, with the Chinese standard's codes beside them
(``AttributesDictionary.csv``, columns ``AHA_code`` and ``CHN_code``).

The AHA codes are bridged to the rotation's classes by the same table that
bridges Shandong's, Leinonen et al.'s AHA-to-SNOMED mapping, read on the base
code (``I106`` is AHA 106).  A child's ECG differs from an adult's in rate,
axis and intervals, so ZZU is read only as a target of the adult models and
never pooled with the adult corpora in a summary.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import MAPPINGS_DIR
from .mimic import one_per_patient
from .small_set import Ambiguity, aha_codes_for

__all__ = [
    "AMBIGUITIES",
    "IAVB_STATEMENT",
    "REFUSED",
    "ZZU_DEVIATIONS",
    "ZZU_SEED",
    "aha_tokens",
    "zzu_frame",
    "zzu_labels",
]

REFUSED: tuple[tuple[str, str], ...] = (("zzu", "NSR"),)

# The release writes a statement the AHA standard has no code for as the
# statement itself, in place of the code.
IAVB_STATEMENT = "First-degree AV block"

ZZU_SEED = 27  # the rotation's split seed

# An AHA code as the release writes it: a category letter and the code number,
# with an optional modifier ("D30+Modifier310").  "Suggests208" and
# "Consider221" are statements of another kind and do not match.
_CODE = re.compile(r"^[A-N](\d+)$")

AMBIGUITIES: tuple[Ambiguity, ...] = (
    Ambiguity(
        key="zzu-no-sinus-rhythm-code",
        what=(
            "ZZU's 91 statements include sinus tachycardia, bradycardia and arrhythmia, "
            "'Normal ECG' (AHA A1) and 'Otherwise normal ECG' (A2), but no statement of "
            "sinus rhythm itself (Supplementary Table 1 of Tan et al. 2025 lists none)."
        ),
        corpora=("zzu",),
        decision=(
            "NSR is refused on ZZU, as on Shandong and CODE, rather than filled from "
            "'Normal ECG' or from the absence of every other rhythm statement."
        ),
        why=(
            "'Normal ECG' is a statement about the whole tracing and covers 1,906 of "
            "14,190 records; reading it as sinus rhythm would put a far narrower class "
            "on the children than on the adults they are compared with."
        ),
        evidence=(
            "zzu-pecg/ECGCode.csv has no sinus-rhythm row; Supplementary Table 1 of "
            "doi:10.1038/s41597-025-05225-z, rows A1, A2, C21 to C26"
        ),
        refuses=REFUSED,
    ),
    Ambiguity(
        key="zzu-iavb-without-aha-code",
        what=(
            "ZZU writes first-degree AV block two ways: 165 records carry the Chinese "
            "standard's 'First-degree AV block' (CHN H74), which has no AHA code, so the "
            "AHA column holds the statement's text; 12 carry AHA H82 'Prolonged PR "
            "interval', which Leinonen's table maps to first-degree block."
        ),
        corpora=("zzu",),
        decision=(
            "Both count as IAVB: the AHA base code 82 through Leinonen's table, and the "
            "statement text 'First-degree AV block' where the code is missing."
        ),
        why=(
            "The rotation already unions the two names on every corpus (ambiguity "
            "'iavb-lpr'); reading the code alone would keep 12 of 177 ZZU records and "
            "make the class mean something else on the children."
        ),
        evidence=(
            "Supplementary Table 1 of doi:10.1038/s41597-025-05225-z: 'First-degree AV "
            "block, First-degree AV block, H74, 165' and 'Prolonged PR interval, H82, 12'"
        ),
    ),
)

ZZU_DEVIATIONS = (
    "children aged 0 to 14, inpatients; a side arm read only as a target",
    "twelve-lead records only, one per patient drawn with a fixed seed, all in the test part",
    "records run from five to 120 seconds; the first ten are kept, shorter ones dropped",
    "statements by the ECG room's physicians, converted to AHA codes by the release",
)


def aha_tokens(raw: str) -> set[str]:
    """The release's AHA column split into its entries, quotes stripped."""
    return {token.strip().strip("'").strip() for token in str(raw).split(";") if token.strip()}


def _base_codes(tokens: set[str]) -> set[str]:
    out = set()
    for token in tokens:
        match = _CODE.match(token.split("+", 1)[0])
        if match:
            out.add(match.group(1))
    return out


def zzu_labels(aha_column: pd.Series, root: Path = MAPPINGS_DIR) -> pd.DataFrame:
    """One boolean column per rotation class for each ZZU record; NSR refused."""
    tokens = aha_column.apply(aha_tokens)
    base = tokens.apply(_base_codes)
    columns = {"NSR": pd.Series(False, index=aha_column.index)}
    for key in ("AF", "LBBB", "RBBB", "IAVB"):
        wanted = aha_codes_for(key, root)
        columns[key] = base.apply(lambda codes, want=wanted: bool(want & codes))
    columns["IAVB"] = columns["IAVB"] | tokens.apply(lambda t: IAVB_STATEMENT in t)
    return pd.DataFrame(columns, index=aha_column.index)


def _years(age: str) -> float:
    """The release writes age in days, as ``572d``."""
    text = str(age).strip()
    return float(text.removesuffix("d")) / 365.25 if text.endswith("d") else float("nan")


def zzu_frame(root: Path, records: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One twelve-lead record of at least ten seconds per child, and its labels.

    ``root`` holds ``AttributesDictionary.csv``; ``records`` is where
    ``Child_ecg/`` was unpacked from the split archive.
    """
    table = pd.read_csv(root / "AttributesDictionary.csv")
    table.index = pd.Index(table["ECG_ID"].astype(str))
    table = table[(table["Lead"] == 12) & (table["Sampling_point"] >= 5000)]
    table = table.loc[one_per_patient(table["Patient_ID"].astype(str), ZZU_SEED)]
    frame = pd.DataFrame(
        {
            "patient": table["Patient_ID"].astype(str),
            "path": [str(records / "Child_ecg" / f) for f in table["Filename"]],
            "age": [_years(a) for a in table["Age"]],
            "sex": np.where(table["Gender"].str.contains("Female"), "female", "male"),
        },
        index=table.index,
    )
    return frame, zzu_labels(table["AHA_code"])

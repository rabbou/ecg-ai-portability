"""CODE-15% and the CODE test set read as two more corpora: Brazilian telehealth.

Both come from the Telehealth Network of Minas Gerais (TNMG), which reads the
ECGs of 811 of the state's 853 municipalities, and both carry the same six
flags per exam: first-degree AV block (``1dAVb``), right and left bundle-branch
block, sinus bradycardia and tachycardia, and atrial fibrillation.

CODE-15% (Zenodo 4916206, CC BY 4.0) holds 345,779 exams from 233,770 patients,
2010-2016, "obtained through stratified sampling from the CODE dataset (15% of
the patients)".  Its flags are the CODE dataset's: a cardiologist's free-text
report, the Uni-G automatic statements and Minnesota codes, reconciled by rules
on the measured intervals, with about 34,512 exams reviewed by hand (Ribeiro et
al., Nat Commun 2020, Methods).  They are not a reading by two cardiologists.

The CODE test set (Zenodo 3765780, CC BY 4.0) holds 827 exams from 827
patients, April to September 2018, each annotated by two cardiologists, a
third settling disagreements (its ``README.md``; Ribeiro et al. 2020).  Its
``gold_standard.csv`` is that consensus.

Both ship their tracings as (n, 4096, 12) at 400 Hz: ten-second exams (4,000
samples) and seven-second ones (2,800 to 2,934) padded with zeros on both sides.
The rotation reads ten seconds.  CODE-15% is sampled among its ten-second
exams only.  The test set is taken whole, as its own labels are the point of
it, and its short exams are zero-filled to ten seconds, a deviation named
wherever its rows appear.

No CODE flag is sinus rhythm: ``normal_ecg`` is the automatic system's "normal
ECG", narrower than the class, the same refusal as Shandong's "Normal ECG".
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .ingest import Record
from .small_set import Ambiguity

__all__ = [
    "AMBIGUITIES",
    "CODE15_DEVIATIONS",
    "CODE15_SAMPLE_SEED",
    "CODE15_SAMPLE_SIZE",
    "CODE_FLAGS",
    "CODE_TEST_DEVIATIONS",
    "CODE_TEST_LEADS",
    "CODE15_LEADS",
    "REFUSED",
    "TEN_SECONDS_AT_400",
    "code15_frame",
    "code15_pool",
    "code15_sample",
    "code_labels",
    "code_test_frame",
    "proportional_allocation",
    "read_code15",
    "read_code_test",
    "short_exams",
    "spans_of",
    "strata_of",
]

# The rotation's class key to the CODE flag that carries it.
CODE_FLAGS: dict[str, str] = {"AF": "AF", "LBBB": "LBBB", "RBBB": "RBBB", "IAVB": "1dAVb"}

REFUSED: tuple[tuple[str, str], ...] = (("code15", "NSR"), ("code_test", "NSR"))

TEN_SECONDS_AT_400 = 4000

# Lead order of each release, from its own description, and checked against
# Einthoven's and Goldberger's relations to leads I and II (aVR = -(I+II)/2,
# aVL = I - II/2, aVF = II - I/2) on the tracings themselves: on the test set the
# columns 3, 4, 5 match aVL, aVF and aVR with a median relative residual of 2 to
# 3%, and any other assignment leaves more than 59%.  The two releases differ.
CODE15_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
CODE_TEST_LEADS = ("I", "II", "III", "aVL", "aVF", "aVR", "V1", "V2", "V3", "V4", "V5", "V6")
SAMPLING_RATE_HZ = 400

# The sample: enough patients that the rotation's caps fill (a test part of 8,000
# at a 40% share needs 20,000), drawn from the ten-second exams.
CODE15_SAMPLE_SIZE = 22_000
CODE15_SAMPLE_SEED = 27  # the rotation's split seed

AMBIGUITIES: tuple[Ambiguity, ...] = (
    Ambiguity(
        key="code-normal-versus-sinus",
        what=(
            "CODE carries no sinus-rhythm flag. Its 'normal_ecg' column is the automatic "
            "system's statement that the whole tracing is normal, which the Zenodo record "
            "describes as 'True if automatic annotation system say it is a normal ECG'. "
            "The rotation's NSR class is sinus rhythm on any tracing, abnormal or not."
        ),
        corpora=("code15", "code_test"),
        decision=(
            "NSR is refused on both CODE corpora, as on Shandong, and the class leaves "
            "their rows rather than being filled from 'normal_ecg' or from the absence "
            "of the other rhythm flags."
        ),
        why=(
            "38.9% of CODE-15% exams are 'normal_ecg', against 83% sinus rhythm in "
            "PTB-XL; reading one as the other would change what the class means on one "
            "corpus, which the rotation could not tell from a change of hospital."
        ),
        evidence=(
            "code15/record.json description of exams.csv; code-test data/README.md lists "
            "six annotations, none of them sinus rhythm"
        ),
        refuses=REFUSED,
    ),
    Ambiguity(
        key="code-test-short-exams",
        what=(
            "539 of the CODE test set's 827 exams hold 7.3 to 7.8 seconds of signal "
            "(2,934 or 3,134 samples at 400 Hz) and are padded to 4,096 with zeros, as the "
            "release's README says. The rotation's models read ten seconds and were "
            "trained on ten-second tracings."
        ),
        corpora=("code_test",),
        decision=(
            "The test set is kept whole: a short exam is cut to its signal and filled "
            "with zeros to ten seconds at the end. The ten-second exams are also "
            "reported on their own, so the effect of the fill can be read."
        ),
        why=(
            "Dropping the short exams would keep 288 of 827 and 5 of the 13 atrial "
            "fibrillations; the cardiologist consensus is what makes the set worth "
            "reading, so the exams stay and the fill is named."
        ),
        evidence=(
            "first and last non-zero sample of each exam in ecg_tracings.hdf5: 538 spans "
            "of 2,934 samples, 253 of 4,096, 35 of 4,000, 1 of 3,134"
        ),
    ),
    Ambiguity(
        key="code-labels-are-reconciled",
        what=(
            "CODE-15%'s flags were not read by two cardiologists. Ribeiro et al. 2020 "
            "build them from the TNMG cardiologist's report, the Uni-G statements and "
            "Minnesota codes, with interval rules (RBBB and LBBB refused under a QRS of "
            "115 ms, first-degree block under a PR of 190 ms) and a hand review of the "
            "exams the rules left open."
        ),
        corpora=("code15",),
        decision=(
            "The flags are used as published. Every CODE-15% row is labelled as "
            "reconciled report labels, and the test set's cardiologist consensus is "
            "kept as a separate corpus so the two kinds of label are never pooled."
        ),
        why=(
            "The rotation compares thresholds across hospitals that label differently; "
            "naming how each corpus labels is how that difference stays readable, as "
            "MIMIC's machine labels are named in MIMIC.md."
        ),
        evidence=(
            "Ribeiro et al., Nat Commun 11, 1760 (2020), Methods, 'Labeling training "
            "data'; Zenodo 4916206 description"
        ),
    ),
)

CODE15_DEVIATIONS = (
    "a stratified sample of 22,000 patients, one ten-second exam each, from CODE-15%'s "
    "345,779 exams; the labels are CODE's reconciled report labels",
    "sampled at 400 Hz and resampled to 500 Hz",
    "lead order from the release description, checked against the limb-lead relations",
    "age and sex from exams.csv",
)

CODE_TEST_DEVIATIONS = (
    "the 827 exams whole, all in the test part; labels are the two-cardiologist consensus",
    "539 exams of 7.3 to 7.8 s cut to their signal and zero-filled to ten seconds",
    "sampled at 400 Hz and resampled to 500 Hz",
    "lead order I, II, III, aVL, aVF, aVR, V1-V6, as the release's README states and the "
    "limb-lead relations confirm",
)


def code_labels(flags: pd.DataFrame) -> pd.DataFrame:
    """One boolean column per rotation class; NSR all False, refused."""
    columns = {"NSR": pd.Series(False, index=flags.index)}
    for key, flag in CODE_FLAGS.items():
        # exams.csv writes True/False, gold_standard.csv writes 1/0.
        columns[key] = flags[flag].map(lambda v: str(v).strip().lower() in {"true", "1"})
    return pd.DataFrame(columns, index=flags.index)[["NSR", "AF", "LBBB", "RBBB", "IAVB"]]


def spans_of(tracings: NDArray[np.floating]) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """First and one-past-last sample where any lead is non-zero, per exam of (n, T, 12)."""
    live = np.abs(tracings).sum(axis=2) > 0
    first = live.argmax(axis=1)
    last = tracings.shape[1] - live[:, ::-1].argmax(axis=1)
    empty = ~live.any(axis=1)
    first[empty] = 0
    last[empty] = 0
    return first, last


def proportional_allocation(sizes: pd.Series, n: int) -> pd.Series:
    """How many to draw from each stratum so the draw keeps the strata's shares.

    Largest-remainder rounding: the floors of ``n * share``, then one more for the
    strata with the largest remainders until the total is ``n``; ties go to the
    stratum listed first.
    """
    if n > int(sizes.sum()):
        raise ValueError(f"cannot draw {n} from {int(sizes.sum())}")
    exact = sizes / sizes.sum() * n
    base = np.floor(exact).astype(int)
    short = n - int(base.sum())
    remainder = (exact - base).to_numpy()
    order = np.argsort(-remainder, kind="stable")[:short]
    base.iloc[order] += 1
    return base


def code15_pool(
    exams: pd.DataFrame, spans: pd.Series, seed: int = CODE15_SAMPLE_SEED
) -> pd.DataFrame:
    """One ten-second exam per patient who has one, drawn with ``seed``.

    ``exams`` is ``exams.csv`` and ``spans`` the non-zero span of every exam,
    indexed by ``exam_id``.  Returns rows of ``exams`` indexed by ``exam_id``.
    """
    exams = exams.set_index("exam_id", drop=False).sort_index()
    eligible = exams[spans.reindex(exams.index).fillna(0).to_numpy() >= TEN_SECONDS_AT_400]
    shuffled = eligible.iloc[np.random.default_rng(seed).permutation(len(eligible))]
    return shuffled[~shuffled["patient_id"].duplicated(keep="first")].sort_index()


def strata_of(rows: pd.DataFrame) -> pd.Series:
    """Each row's combination of the four rotation flags, as a string like ``0100``."""
    return code_labels(rows)[list(CODE_FLAGS)].astype(int).astype(str).agg("".join, axis=1)


def code15_sample(
    exams: pd.DataFrame,
    spans: pd.Series,
    n: int = CODE15_SAMPLE_SIZE,
    seed: int = CODE15_SAMPLE_SEED,
) -> pd.DataFrame:
    """``n`` patients of :func:`code15_pool`, stratified on the four flags.

    The patients are drawn within strata of the combination of the four rotation
    flags, in proportion to each stratum's size, so the sample's prevalences are
    the pool's to within one patient per stratum.
    """
    pool = code15_pool(exams, spans, seed)
    stratum = strata_of(pool)
    quota = proportional_allocation(stratum.value_counts().sort_index(), n)
    rng = np.random.default_rng(seed + 1)
    chosen: list[int] = []
    for name, k in quota.items():
        members = stratum.index[stratum == name].to_numpy()
        chosen.extend(int(e) for e in rng.choice(members, size=int(k), replace=False))
    return pool.loc[sorted(chosen)]


def code15_frame(root: Path, sample: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The sampled CODE-15% exams as rotation rows, and their labels.

    ``root`` holds ``exams.csv``; ``sample`` is the HDF5 file
    ``scripts/code15_extract.py extract`` wrote.  ``path`` is the row in it.
    """
    exams = pd.read_csv(root / "exams.csv").set_index("exam_id", drop=False)
    with h5py.File(sample, "r") as f:
        ids = f["exam_id"][()]
    rows = exams.loc[ids]
    index = pd.Index([str(int(e)) for e in ids])
    frame = pd.DataFrame(
        {
            "patient": rows["patient_id"].astype(str).to_numpy(),
            "path": [f"{sample}#{i}" for i in range(len(ids))],
            "age": rows["age"].to_numpy(dtype=float),
            "sex": np.where(rows["is_male"].astype(bool), "male", "female"),
        },
        index=index,
    )
    labels = code_labels(rows)
    labels.index = index
    return frame, labels


def code_test_frame(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The 827 CODE test exams as rotation rows, one patient each, and their labels."""
    data = root / "data"
    attributes = pd.read_csv(data / "attributes.csv")
    gold = pd.read_csv(data / "annotations" / "gold_standard.csv")
    index = pd.Index([f"code_test_{i:03d}" for i in range(len(attributes))])
    frame = pd.DataFrame(
        {
            "patient": list(index),
            "path": [f"{data / 'ecg_tracings.hdf5'}#{i}" for i in range(len(attributes))],
            "age": attributes["age"].to_numpy(dtype=float),
            "sex": np.where(attributes["sex"] == "M", "male", "female"),
        },
        index=index,
    )
    labels = code_labels(gold)
    labels.index = index
    return frame, labels


@cache
def _open(path: str) -> h5py.File:
    return h5py.File(path, "r")


def _row(path: Path | str) -> tuple[h5py.File, int]:
    """The open release file and the row a ``<file>#<row>`` path names."""
    file, _, row = str(path).rpartition("#")
    return _open(file), int(row)


def read_code15(path: Path | str) -> Record:
    """One sampled CODE-15% exam: ten seconds at 400 Hz, already cut by the extractor."""
    f, row = _row(path)
    signal = np.ascontiguousarray(f["tracings"][row].T, dtype=np.float32)
    return Record(signal, CODE15_LEADS, SAMPLING_RATE_HZ)


def read_code_test(path: Path | str) -> Record:
    """One CODE test exam, cut to its signal and zero-filled to ten seconds at the end."""
    f, row = _row(path)
    tracing = np.asarray(f["tracings"][row], dtype=np.float32)
    first, last = (int(v[0]) for v in spans_of(tracing[None]))
    signal = tracing[first:last][:TEN_SECONDS_AT_400]
    filled = np.zeros((TEN_SECONDS_AT_400, signal.shape[1]), dtype=np.float32)
    filled[: len(signal)] = signal
    return Record(np.ascontiguousarray(filled.T), CODE_TEST_LEADS, SAMPLING_RATE_HZ)


def short_exams(paths: Sequence[Path | str]) -> list[bool]:
    """Whether each CODE test exam holds less than ten seconds of signal."""
    out = []
    for path in paths:
        f, row = _row(path)
        first, last = (int(v[0]) for v in spans_of(np.asarray(f["tracings"][row])[None]))
        out.append(last - first < TEN_SECONDS_AT_400)
    return out

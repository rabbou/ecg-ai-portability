"""Read CODE-15% where it lies: the length of every exam, then the sampled exams.

CODE-15% ships 345,779 exams as eighteen zipped HDF5 files of about 2.7 GB
each, ``tracings`` of shape (n, 4096, 12) at 400 Hz.  An exam of ten seconds
holds 4,000 samples and one of seven holds 2,800; both are padded with zeros
on either side to 4,096 (the Zenodo record's description).  The rotation reads
ten seconds, so which exams are long enough has to be known before the sample
is drawn, and that needs every exam read once.

Two passes, each part decompressed to ``--work`` one at a time and deleted
before the next, so the machine holding the zips needs about 4 GB free beside
them:

``spans``     one row per exam: its part, the index of its first and last
              non-zero sample, and the span between them; plus, per part, the
              check that columns 3 to 6 are the leads the description names
              (aVR, aVL, aVF are fixed by Einthoven's and Goldberger's
              relations to I and II).
``extract``   the exams named in ``--ids``, each cut to the 4,000 samples from
              its first non-zero one, written as ``exam_id`` (n,) and
              ``tracings`` (n, 4000, 12) float32 to one HDF5 file.

Needs numpy and h5py only, so the box runs it without the project:

    uv run --no-project --with h5py --with numpy python code15_extract.py spans \\
        --zips ~/data/code15 --work ~/tmp-code15 --out ~/tmp-code15/spans.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import h5py
import numpy as np
from numpy.typing import NDArray

TEN_SECONDS = 4000  # samples at 400 Hz
CHUNK = 1000
# Lead order the Zenodo description gives for exams_part{i}.hdf5.
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")


def parts(zips: Path) -> list[Path]:
    def number(path: Path) -> int:
        return int(path.stem.removeprefix("exams_part"))

    return sorted(zips.glob("exams_part*.zip"), key=number)


def unpacked(archive: Path, work: Path) -> Path:
    """Decompress the one HDF5 member of ``archive`` into ``work``."""
    with zipfile.ZipFile(archive) as z:
        (member,) = [m for m in z.namelist() if m.endswith(".hdf5")]
        target = work / Path(member).name
        with z.open(member) as source, target.open("wb") as sink:
            shutil.copyfileobj(source, sink, length=64 << 20)
    return target


def spans_of(block: NDArray[np.floating]) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """First and one-past-last sample where any lead is non-zero, per exam."""
    live = np.abs(block).sum(axis=2) > 0
    first = live.argmax(axis=1)
    last = block.shape[1] - live[:, ::-1].argmax(axis=1)
    empty = ~live.any(axis=1)
    first[empty] = 0
    last[empty] = 0
    return first, last


def lead_residuals(block: NDArray[np.floating]) -> dict[str, float]:
    """Median relative residual of each limb lead against its value from I and II."""
    lead_i, lead_ii = block[:, :, 0], block[:, :, 1]
    expected = {
        "III": lead_ii - lead_i,
        "aVR": -(lead_i + lead_ii) / 2,
        "aVL": lead_i - lead_ii / 2,
        "aVF": lead_ii - lead_i / 2,
    }
    out = {}
    for name, value in expected.items():
        column = block[:, :, LEADS.index(name)]
        scale = np.abs(value).mean(axis=1) + 1e-9
        out[name] = float(np.median(np.abs(column - value).mean(axis=1) / scale))
    return out


def scan(zips: Path, work: Path, out: Path) -> int:
    work.mkdir(parents=True, exist_ok=True)
    checks: dict[str, dict[str, float]] = {}
    with out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["exam_id", "part", "first", "last", "span"])
        for archive in parts(zips):
            started = time.time()
            path = unpacked(archive, work)
            with h5py.File(path, "r") as f:
                ids = f["exam_id"][()]
                x = f["tracings"]
                for start in range(0, len(ids), CHUNK):
                    block = x[start : start + CHUNK]
                    first, last = spans_of(block)
                    for exam, a, b in zip(ids[start : start + CHUNK], first, last, strict=True):
                        writer.writerow([int(exam), archive.stem, int(a), int(b), int(b - a)])
                    if start == 0:
                        checks[archive.stem] = lead_residuals(block)
            path.unlink()
            print(f"{archive.name}: {len(ids)} exams, {time.time() - started:.0f} s", flush=True)
    (out.with_suffix(".leads.json")).write_text(json.dumps(checks, indent=1) + "\n")
    return 0


def extract(zips: Path, work: Path, spans: Path, ids_file: Path, out: Path) -> int:
    wanted = {int(line) for line in ids_file.read_text().split() if line.strip()}
    where: dict[int, tuple[str, int]] = {}
    with spans.open() as handle:
        for row in csv.DictReader(handle):
            exam = int(row["exam_id"])
            if exam in wanted:
                where[exam] = (row["part"], int(row["first"]))
    missing = wanted - set(where)
    if missing:
        raise SystemExit(f"{len(missing)} requested exams are not in {spans}")
    work.mkdir(parents=True, exist_ok=True)
    order = sorted(wanted)
    slot = {exam: i for i, exam in enumerate(order)}
    with h5py.File(out, "w") as sink:
        sink["exam_id"] = np.asarray(order, dtype=np.int64)
        tracings = sink.create_dataset("tracings", (len(order), TEN_SECONDS, 12), dtype="f4")
        sink.attrs["sampling_rate_hz"] = 400
        sink.attrs["leads"] = list(LEADS)
        written = 0
        for archive in parts(zips):
            here = {e for e, (part, _) in where.items() if part == archive.stem}
            if not here:
                continue
            path = unpacked(archive, work)
            with h5py.File(path, "r") as f:
                ids = f["exam_id"][()]
                rows = np.flatnonzero(np.isin(ids, list(here)))
                for row in rows:
                    exam = int(ids[row])
                    first = where[exam][1]
                    tracings[slot[exam]] = f["tracings"][row, first : first + TEN_SECONDS, :]
                    written += 1
            path.unlink()
            print(f"{archive.name}: {len(rows)} of {written} / {len(order)}", flush=True)
    if written != len(order):
        raise SystemExit(f"wrote {written} of {len(order)} exams")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pass_", choices=("spans", "extract"), metavar="spans|extract")
    parser.add_argument("--zips", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--spans", type=Path, help="extract: the file the spans pass wrote")
    parser.add_argument("--ids", type=Path, help="extract: one exam_id per line")
    args = parser.parse_args()
    if args.pass_ == "spans":
        return scan(args.zips, args.work, args.out)
    if args.spans is None or args.ids is None:
        parser.error("extract needs --spans and --ids")
    return extract(args.zips, args.work, args.spans, args.ids, args.out)


if __name__ == "__main__":
    sys.exit(main())

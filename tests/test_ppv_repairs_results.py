"""The committed PPV-repair results, held to the scores they came from and to PPV.md.

The targets v of every transfer whose scores this repository carries are
recomputed here from the source calibration scores alone, without importing
``ecs``: the conformal rank and the share of ill among the flagged are written
out again, so agreement says the file's v were fixed from the source.  The
EchoNext targets need restricted scores, and are rebuilt under ``-m data``.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

import numpy as np
import pytest

from ecs.config import REPO_ROOT, RESULTS_DIR

ALPHA = 0.10
FILE = RESULTS_DIR / "ppv_repairs.json"
RESERVE = RESULTS_DIR / "ppv_repairs_reserve.json"
MAIN = ("columbia", "beth_israel", "chongqing")
CLASSES = ("NSR", "AF", "LBBB", "RBBB", "IAVB")


@pytest.fixture(scope="module")
def result() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(FILE.read_text())
    return loaded


def source_v(p: np.ndarray, y: np.ndarray) -> float:
    """Share of ill among the flagged at the ceil((n+1)(1-alpha))-th smallest ill score 1 - p."""
    scores = np.sort(1.0 - p[y == 1])
    rank = math.ceil((len(scores) + 1) * (1 - ALPHA))
    flagged = (1.0 - p) <= scores[rank - 1]
    return float(y[flagged].mean())


def source_scores(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if row["family"] == "infarction":
        with np.load(RESULTS_DIR / "baseline/scores.npz") as s:
            return s["probs"][:, 1].astype(float), s["labels"].astype(int)
    k = CLASSES.index(row["label"])
    path = RESULTS_DIR / "rotation_seven" / row["source"] / "scores" / f"{row['source']}_cal.npz"
    with np.load(path) as s:
        return s["p"][:, k].astype(float), s["y"][:, k].astype(int)


@pytest.fixture(scope="module")
def reserve() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(RESERVE.read_text())
    return loaded


def test_every_public_target_is_the_source_ppv(
    result: dict[str, Any], reserve: dict[str, Any]
) -> None:
    """Protocol section 7: each v is the source's PPV at its own threshold, before any
    evaluation patient is read; a v drawn from the target would grade every repair
    against the answer."""
    rows = result["rows"] + reserve["rows"]
    public = [r for r in rows if r["family"] in ("infarction", "rotation_seven")]
    assert len(public) > 200
    for row in public:
        p, y = source_scores(row)
        assert row["v"] == pytest.approx(source_v(p, y), rel=1e-3), row


@pytest.mark.data
def test_every_echonext_target_is_frozen_from_the_source(result: dict[str, Any]) -> None:
    """The EchoNext v rebuilt with every target array blanked: none moves."""
    from ppv_repairs import freeze_targets, key_of
    from score_pairs import Pair, echonext, echonext_available

    if not echonext_available():
        pytest.skip("EchoNext and its stored scores are not on this machine")
    blind = [
        Pair(
            p.family,
            p.model,
            p.source,
            p.target,
            p.label,
            p.p_source,
            p.y_source,
            np.full_like(p.p_target, np.nan),
            np.zeros_like(p.y_target),
            None,
        )
        for p in echonext()
    ]
    frozen = freeze_targets(blind)
    rows = [r for r in result["rows"] if r["family"] == "echonext"]
    assert rows
    for row in rows:
        key = "/".join(row[k] for k in ("family", "model", "source", "target", "label"))
        assert row["v"] == pytest.approx(frozen[key].ppv, rel=1e-3), key
    assert key_of(blind[0]) in frozen


def test_every_hospital_summary_counts_its_rows(
    result: dict[str, Any], reserve: dict[str, Any]
) -> None:
    """A summary filter that drifted from the rows would print a count no row backs."""
    every = result["rows"] + reserve["rows"]
    assert set(result["summary"]) == {r["hospital"] for r in every}
    for hospital in result["summary"]:
        rows = [r for r in every if r["hospital"] == hospital and r["summarised"]]
        assert result["summary"][hospital]["transfers"] == len(rows) > 0, hospital
    assert {r["hospital"] for r in result["rows"]} == {*MAIN, "control"}
    assert sum(r["hospital"] == "chongqing" for r in result["rows"]) == 1


def test_the_prior_summary_is_the_median_of_its_rows(result: dict[str, Any]) -> None:
    for hospital in MAIN:
        errors = [
            r["prior"]["abs_error"]
            for r in result["rows"]
            if r["hospital"] == hospital and r["summarised"]
        ]
        block = result["summary"][hospital]["prior"]["abs_error"]
        assert block["median"] == pytest.approx(float(np.median(errors)), rel=1e-3), hospital


def test_the_falsifiers_follow_from_their_numbers(result: dict[str, Any]) -> None:
    """Each falsifier's verdict, decided again from the figures it prints."""
    first = result["falsifiers"]["label_free_outside_columbia"]
    assert first["fired"] == all(u < 0.02 for u in first["upper_bounds"].values())
    second = result["falsifiers"]["certification_at_100"]
    assert second["fired_violation"] == any(v > 0.15 for v in second["violation_rate"].values())
    assert second["fired_existence"] == any(
        s < 0.5 for s in second["share_threshold_exists"].values()
    )


def lists_in(value: Any) -> list[int]:
    if isinstance(value, dict):
        return [n for v in value.values() for n in lists_in(v)]
    if isinstance(value, list):
        return [len(value), *(n for v in value for n in lists_in(v))]
    return []


def test_no_row_carries_per_record_values(result: dict[str, Any]) -> None:
    """EchoNext's licence: no per-patient score may enter the repository."""
    rows = [r for r in result["rows"] if r["family"] == "echonext"]
    assert rows
    assert max(n for r in rows for n in lists_in(r)) <= len(result["budgets"])


def ppv_md_section() -> str:
    text = (REPO_ROOT / "PPV.md").read_text()
    return text[text.index("\n## ") :]


def pts(x: float) -> str:
    return f"{100 * x:.1f}"


def ppv_md_figures(result: dict[str, Any]) -> list[str]:
    """Every figure PPV.md prints, formatted from the file."""
    s = result["summary"]
    out = []
    for h in MAIN:
        prior = s[h]["prior"]
        out.append(f"{pts(prior['abs_error']['median'])} points")
        out.append(f"{s[h]['transfers']} transfer")
        at = {b["n"]: b for b in s[h]["budgets"]}
        for n in (25, 100, 400):
            if at[n]["transfers"]:
                out.append(f"{pts(at[n]['recalibrated_abs_error']['median'])} points")
                out.append(f"{pts(at[n]['resampled_abs_error']['median'])} points")
        out.append(f"{at[100]['certified_violation_rate']:.3f}")
        out.append(f"{at[100]['certified_share_threshold_exists']:.3f}")
        out.append(f"{s[h]['selection']['mean_fdp']:.2f}")
        out.append(f"{s[h]['selection']['mean_q']:.2f}")
    for h in MAIN[1:]:
        upper = result["falsifiers"]["label_free_outside_columbia"]["upper_bounds"][h]
        out.append(f"{pts(upper)} points")
    out.append(f"{pts(s['control']['prior']['abs_error']['median'])} points")
    out.append(f"{pts(result['published_control_median_gap'])} points")
    return out


def test_ppv_md_prints_the_file(result: dict[str, Any]) -> None:
    """A figure retyped by hand drifts from the file it quotes; each is formatted here."""
    section = ppv_md_section()
    missing = [f for f in ppv_md_figures(result) if f not in section]
    assert not missing, missing


def test_ppv_md_prints_no_figure_the_file_does_not_hold(result: dict[str, Any]) -> None:
    """The twin: every 'x.y points' in the section is one of the file's figures."""
    printed = set(re.findall(r"\d+\.\d points", ppv_md_section()))
    assert printed
    assert printed <= set(ppv_md_figures(result)), printed - set(ppv_md_figures(result))

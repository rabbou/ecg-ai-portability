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
CONTROLS = RESULTS_DIR / "ppv_repairs_controls.json"
RESERVE = RESULTS_DIR / "ppv_repairs_reserve.json"
RESERVE_CODE = RESULTS_DIR / "ppv_repairs_reserve_code_zzu.json"
# Rows are written to three significant digits.
ROUNDING = 5e-3
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
def every_row(result: dict[str, Any]) -> list[dict[str, Any]]:
    """The main hospitals' rows, then the controls' and the reserve's from their files."""
    side = [json.loads(path.read_text()) for path in (CONTROLS, RESERVE, RESERVE_CODE)]
    for block in side:
        assert block["targets_digest"] == result["targets_digest"]
        assert block["provenance"] == result["provenance"]
    return [*result["rows"], *(r for block in side for r in block["rows"])]


def test_every_public_target_is_the_source_ppv(every_row: list[dict[str, Any]]) -> None:
    """Protocol section 7: each v is the source's PPV at its own threshold, before any
    evaluation patient is read; a v drawn from the target would grade every repair
    against the answer."""
    public = [r for r in every_row if r["family"] in ("infarction", "rotation_seven")]
    assert len(public) > 200
    for row in public:
        p, y = source_scores(row)
        assert row["v"] == pytest.approx(source_v(p, y), rel=ROUNDING), row


@pytest.mark.data
def test_every_echonext_target_is_frozen_from_the_source(every_row: list[dict[str, Any]]) -> None:
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
    rows = [r for r in every_row if r["family"] == "echonext"]
    assert rows
    for row in rows:
        key = "/".join(row[k] for k in ("family", "model", "source", "target", "label"))
        assert row["v"] == pytest.approx(frozen[key].ppv, rel=ROUNDING), key
    assert key_of(blind[0]) in frozen


def test_every_hospital_summary_counts_its_rows(
    result: dict[str, Any], every_row: list[dict[str, Any]]
) -> None:
    """A summary filter that drifted from the rows would print a count no row backs."""
    every = every_row
    assert set(result["summary"]) == {r["hospital"] for r in every}
    for hospital in result["summary"]:
        rows = [r for r in every if r["hospital"] == hospital and r["summarised"]]
        assert result["summary"][hospital]["transfers"] == len(rows) > 0, hospital
    assert {r["hospital"] for r in result["rows"]} == set(MAIN)
    assert sum(r["hospital"] == "chongqing" for r in result["rows"]) == 1


def test_the_prior_summary_is_the_median_of_its_rows(result: dict[str, Any]) -> None:
    for hospital in MAIN:
        errors = [
            r["prior"]["abs_error"]
            for r in result["rows"]
            if r["hospital"] == hospital and r["summarised"]
        ]
        block = result["summary"][hospital]["prior"]["abs_error"]
        assert block["median"] == pytest.approx(float(np.median(errors)), rel=ROUNDING), hospital


def test_the_falsifiers_follow_from_their_numbers(result: dict[str, Any]) -> None:
    """Each falsifier's verdict, decided again from the figures it prints."""
    first = result["falsifiers"]["label_free_outside_columbia"]
    assert first["fired"] == all(u < 0.02 for u in first["upper_bounds"].values())
    second = result["falsifiers"]["certification_at_100"]
    assert second["fired_violation"] == any(v > 0.15 for v in second["violation_rate"].values())
    assert second["fired_existence"] == any(
        s < 0.5 for s in second["share_threshold_exists"].values()
    )


def test_each_label_budget_reports_what_was_announced_and_observed(
    every_row: list[dict[str, Any]],
) -> None:
    """Section 7 asks for the announced and the observed PPV per budget, not their gap alone:
    the gap's size cannot say whether a repair over- or under-announces."""
    blocks = [b[m] for r in every_row for b in r["budgets"] for m in ("recalibrated", "resampled")]
    defined = [b for b in blocks if b["abs_error"] is not None]
    assert len(defined) > 3000
    for b in defined:
        assert 0.0 <= b["announced"] <= 1.0 and 0.0 <= b["observed"] <= 1.0, b
        # |median(x)| <= median(|x|) for any sample.
        assert abs(b["signed_error"]) <= b["abs_error"] + 1e-3, b


def shares_over(result: dict[str, Any], small: bool) -> list[float]:
    """Share of draws over-announcing, per hospital, budget and labelled repair."""
    return [
        b[f"{m}_share_over_announced"]
        for h in MAIN
        for b in result["summary"][h]["budgets"]
        for m in ("recalibrated", "resampled")
        if (b["n"] < 100) == small and (m == "recalibrated" or not small)
    ]


def test_the_direction_ppv_md_states_is_the_file_s(result: dict[str, Any]) -> None:
    """PPV.md: label-free, less than observed in every Columbia transfer, more at Chongqing;
    from 100 labels on, misses on both sides in about equal shares."""
    s = result["summary"]
    assert s["columbia"]["prior"]["share_over_announced"] == 0.0
    assert s["chongqing"]["prior"]["share_over_announced"] == 1.0
    assert all(0.4 <= x <= 0.6 for x in shares_over(result, small=False))
    large = shares_over(result, small=False)
    printed = numbers_in(ppv_md_section())
    for x in (min(large), max(large), max(shares_over(result, small=True))):
        assert pct(x) in printed


def lists_in(value: Any) -> list[int]:
    if isinstance(value, dict):
        return [n for v in value.values() for n in lists_in(v)]
    if isinstance(value, list):
        return [len(value), *(n for v in value for n in lists_in(v))]
    return []


def test_no_row_carries_per_record_values(
    result: dict[str, Any], every_row: list[dict[str, Any]]
) -> None:
    """EchoNext's licence: no per-patient score may enter the repository."""
    rows = [r for r in every_row if r["family"] == "echonext"]
    assert rows
    assert max(n for r in rows for n in lists_in(r)) <= len(result["budgets"])


def ppv_md_section() -> str:
    text = (REPO_ROOT / "PPV.md").read_text()
    return text[text.index("\n## ") :]


def numbers_in(text: str) -> set[str]:
    """Every figure the prose prints, as written: 0.79, 99%, −0.001, 272.

    Table rows are left out, so a figure the tables hold cannot stand in for the prose.
    """
    prose = "\n".join(line for line in text.splitlines() if not line.startswith("|"))
    return set(re.findall(r"−?\d+(?:\.\d+)?%?", prose))


def missing_from(text: str, figures: list[str]) -> list[str]:
    printed = numbers_in(text)
    return [f for f in figures if f not in printed]


def pts(x: float) -> str:
    return f"{100 * x:.1f}"


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def pct1(x: float) -> str:
    """A share of draws known to three decimals, as a percentage: 16.8%, 1.0%, 100%."""
    return "100%" if round(x, 3) == 1 else f"{100 * x:.1f}%"


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
            # Listed as "6.0, 5.7 and 6.6 points", so the unit is not beside each figure.
            out.append(pts(at[n]["recalibrated_abs_error"]["median"]))
            out.append(pts(at[n]["resampled_abs_error"]["median"]))
        out.append(pct1(at[100]["certified_violation_rate"]))
        out.append(pct1(at[100]["certified_share_threshold_exists"]))
        out.append(f"{s[h]['selection']['mean_fdp']:.2f}")
        out.append(f"{s[h]['selection']['mean_q']:.2f}")
        out.append(pct(s[h]["prior"]["sensitivity"]))
        out.append(pct(s[h]["selection"]["sensitivity"]))
        out.append(pct(at[100]["sensitivity"]["certified"]))
        for n in (100, 400):
            out.append(pct(at[n]["sensitivity"]["resampled"]))
        out.append(pts(at[100]["resampled_finite_abs_error"]["median"]))
        # PPV.md: recalibration keeps the source threshold, so its sensitivity is repair 1's.
        assert {pct(b["sensitivity"]["recalibrated"]) for b in at.values()} == {
            pct(prior["sensitivity"])
        }, h
    for h in MAIN[1:]:
        upper = result["falsifiers"]["label_free_outside_columbia"]["upper_bounds"][h]
        out.append(f"{pts(upper)} points")
    out.append(f"{pts(s['control']['prior']['abs_error']['median'])} points")
    # PPV.md: "12.2 to 93.3 points at the eight reserve sites"; the upper end carries the unit.
    reserve = [
        v["prior"]["abs_error"]["median"] for h, v in s.items() if h not in (*MAIN, "control")
    ]
    out.append(f"{pts(max(reserve))} points")
    control = s["control"]["prior"]["abs_error_true_prevalence"]["median"]
    out.append(f"{pts(control)} points")
    out.append(f"{pts(result['published_control_median_gap'])} points")
    published = json.loads((RESULTS_DIR / "echonext_ppv_gap.json").read_text())
    free = published["summary"]["echonext"]["transfer"]["label_free_predictors"]
    out.append(f"{pts(free['recipe_estimated_prevalence']['median_abs_points'])} points")
    return out


def test_ppv_md_prints_the_file(result: dict[str, Any]) -> None:
    """A figure retyped by hand drifts from the file it quotes; each is formatted here."""
    section = ppv_md_section()
    missing = [f for f in ppv_md_figures(result) if f not in section]
    assert not missing, missing


def test_ppv_md_prints_no_figure_the_file_does_not_hold(result: dict[str, Any]) -> None:
    """The twin: every figure in points, alone or in a list ending in 'points', is the file's."""
    listed = r"(\d+\.\d)(?=(?:,? (?:and )?\d+\.\d)* points)"
    printed = set(re.findall(listed, ppv_md_section()))
    held = {f.removesuffix(" points") for f in ppv_md_figures(result)}
    assert len(printed) > 20
    assert printed <= held, printed - held


def ppv_md_part(start: str, stop: str) -> str:
    section = ppv_md_section()
    return section[section.index(start) : section.index(stop)]


def ppv_md_abstract() -> str:
    return ppv_md_part("## Abstract", "## 1. Introduction")


def test_ppv_md_follows_the_article_s_plan() -> None:
    """Structured abstract, then an Introduction ending on one objective sentence."""
    section = ppv_md_section()
    heads = ["## Abstract", "## 1. Introduction", "## 2. Methods", "## 3. Results"]
    heads.append("## 4. Discussion")
    assert [section.index(h) for h in heads] == sorted(section.index(h) for h in heads)
    abstract = ppv_md_abstract()
    labels = ["**Background.**", "**Methods.**", "**Results.**", "**Conclusions.**"]
    assert [abstract.index(x) for x in labels] == sorted(abstract.index(x) for x in labels)
    objective = ppv_md_part("## 1. Introduction", "## 2. Methods").strip().split("\n\n")[-1]
    assert objective.startswith("We ")
    assert objective.count(". ") == 0 and objective.endswith(".")
    assert not re.search(r"\d", objective)
    prose = re.sub(r"`[^`]*`|\]\([^)]*\)", "", section)
    assert not re.search(r"\brepairs?\b", prose, re.IGNORECASE)


def test_ppv_md_abstract_prints_the_file(result: dict[str, Any]) -> None:
    """Each figure of the abstract is the file's, formatted here."""
    from ecs.repairs import MIN_EVENTS

    s = result["summary"]
    at100 = {h: next(b for b in s[h]["budgets"] if b["n"] == 100) for h in MAIN}
    held = [
        str(sum(s[h]["transfers"] for h in MAIN)),
        str(min(result["budgets"])),
        str(max(result["budgets"])),
        str(result["draws"]),
    ]
    for h in MAIN:
        b = at100[h]
        held += [
            pts(s[h]["prior"]["abs_error"]["median"]),
            f"{b['mean_cases_labelled']:.1f}",
            pts(b["recalibrated_abs_error"]["median"]),
            pts(b["resampled_abs_error"]["median"]),
            pts(b["resampled_finite_abs_error"]["median"]),
            pct(flags_all_among_defined(b)),
            pct(b["sensitivity"]["resampled"]),
            pct1(b["certified_share_threshold_exists"]),
            pct1(b["certified_violation_rate_given_threshold"]),
        ]
    assert not missing_from(ppv_md_abstract(), held), missing_from(ppv_md_abstract(), held)
    assert MIN_EVENTS == 10
    assert "fewer than ten events" in ppv_md_section()


SITE_NAMES = {
    "columbia": "Columbia",
    "beth_israel": "Beth Israel",
    "chongqing": "Chongqing",
    "shandong": "Shandong",
    "chapman_ningbo": "Chapman-Ningbo",
    "cpsc": "CPSC",
    "georgia": "Georgia",
    "ptbxl": "PTB-XL",
    "code15": "CODE-15%",
    "code_test": "CODE test set",
    "zzu": "ZZU",
    "control": "In-distribution controls",
}
TABLE_BUDGET = 100


def signed(x: float | None) -> str:
    if x is None:
        return "–"
    return f"{x:.3f}".replace("-0.000", "0.000").replace("-", "−")


def with_bounds(median: float, bounds: list[float] | None) -> str:
    if bounds is None:
        return pts(median)
    return f"{pts(median)} ({pts(bounds[0])} to {pts(bounds[1])})"


def flags_all_among_defined(b: dict[str, Any]) -> float:
    """Share of the draws with a refitted threshold in which it flagged every patient.

    Both rates are means over transfers of counts out of the same number of draws,
    and a draw that flags everyone always has a threshold, so the ratio is exact.
    """
    return float(b["resampled_share_flags_all"] / b["resampled_share_defined"])


def error_table(result: dict[str, Any]) -> list[str]:
    """Table 2 of PPV.md, one row per site, formatted from the file."""
    rows = []
    for h, name in SITE_NAMES.items():
        s = result["summary"][h]
        b = next(x for x in s["budgets"] if x["n"] == TABLE_BUDGET)
        prior = s["prior"]
        low = prior.get("median_abs_error_lower_2_5")
        bounds = [low, prior["median_abs_error_upper_97_5"]] if low is not None else None
        cells = [
            name,
            str(s["transfers"]),
            f"{b['mean_cases_labelled']:.1f}",
            with_bounds(prior["abs_error"]["median"], bounds),
            with_bounds(
                b["recalibrated_abs_error"]["median"], b["recalibrated_abs_error_median_bounds"]
            ),
            with_bounds(b["resampled_abs_error"]["median"], b["resampled_abs_error_median_bounds"]),
            pct(b["resampled_share_defined"]),
            pct(flags_all_among_defined(b)),
        ]
        rows.append("| " + " | ".join(cells) + " |")
    return rows


def net_benefit_table(result: dict[str, Any]) -> list[str]:
    """Table 3 of PPV.md, one row per site, formatted from the file.

    Recalibration keeps the source threshold, so its net benefit is the prevalence
    correction's and the two share a column.
    """
    rows = []
    for h, name in SITE_NAMES.items():
        s = result["summary"][h]
        nb = next(x for x in s["budgets"] if x["n"] == TABLE_BUDGET)["net_benefit"]
        assert nb["recalibrated"] == pytest.approx(s["prior"]["net_benefit"], abs=5e-4), h
        cells = [
            name,
            signed(s["references"]["treat_all_net_benefit"]),
            signed(s["references"]["own_cut_net_benefit"]),
            signed(s["prior"]["net_benefit"]),
            signed(nb["resampled"]),
            signed(nb["certified"]),
            signed(s["selection"]["net_benefit"]),
        ]
        rows.append("| " + " | ".join(cells) + " |")
    return rows


BUDGET_SITES = (*MAIN, "control")


def finite_error(b: dict[str, Any]) -> str:
    if not b["resampled_finite_abs_error"].get("n"):
        return "–"
    return with_bounds(
        b["resampled_finite_abs_error"]["median"],
        b["resampled_finite_abs_error_median_bounds"],
    )


def budget_table(result: dict[str, Any]) -> list[str]:
    """Table 1 of PPV.md, one row per site and budget, formatted from the file."""
    rows = []
    for h in BUDGET_SITES:
        for b in result["summary"][h]["budgets"]:
            sens = b["resampled_finite_sensitivity"]
            cells = [
                SITE_NAMES[h],
                str(b["n"]),
                f"{b['mean_cases_labelled']:.1f}",
                pct(b["resampled_share_defined"]),
                pct(flags_all_among_defined(b)),
                pts(b["resampled_abs_error"]["median"]),
                finite_error(b),
                pct(sens) if sens is not None else "–",
            ]
            rows.append("| " + " | ".join(cells) + " |")
    return rows


def test_ppv_md_tables_are_the_file_s(result: dict[str, Any]) -> None:
    """Every row of Tables 1 to 3 is the file's, reserve sites and controls included."""
    section = ppv_md_section()
    rows = budget_table(result) + error_table(result) + net_benefit_table(result)
    missing = [row for row in rows if row not in section]
    assert not missing, missing


def test_ppv_md_budget_claims_are_the_file_s(result: dict[str, Any]) -> None:
    """The prose around Table 1: finite-threshold errors, where flagging everyone falls."""
    s = result["summary"]
    at = {h: {b["n"]: b for b in s[h]["budgets"]} for h in BUDGET_SITES}
    finite = {h: at[h][100]["resampled_finite_abs_error"]["median"] for h in MAIN}
    recal = {h: at[h][100]["recalibrated_abs_error"]["median"] for h in MAIN}
    assert {h for h in MAIN if finite[h] > recal[h]} == {"columbia", "chongqing"}
    below_half = {
        h: min(n for n, b in at[h].items() if flags_all_among_defined(b) < 0.5) for h in MAIN
    }
    assert below_half == {"columbia": 400, "beth_israel": 400, "chongqing": 200}
    nine = {h: s[h]["labels_for_nine_cases_per_half"]["median"] for h in MAIN}
    columbia = [b["certified_violation_rate_given_threshold"] for b in at["columbia"].values()]
    assert min(columbia) > result["delta"]
    assert at["columbia"][25]["resampled_finite_abs_error"]["n"] == 18
    assert at["chongqing"][25]["resampled_finite_abs_error"] == {"n": 0}
    assert at["chongqing"][50]["resampled_finite_abs_error"]["n"] == 3
    caption = (
        f"With 25 labels the Columbia value rests on "
        f"{at['columbia'][25]['resampled_finite_abs_error']['n']} draws and no Chongqing "
        f"threshold was finite; with 50 labels the Chongqing value rests on "
        f"{at['chongqing'][50]['resampled_finite_abs_error']['n']} draws."
    )
    assert caption in ppv_md_section()
    found = at["chongqing"][100]["certified_share_threshold_exists"] * 200
    assert found == pytest.approx(2)
    refitted = ppv_md_part("### 3.2", "### 3.3")
    figures = [f"{nine[h]:.0f}" for h in MAIN]
    for h in MAIN:
        figures += [pts(finite[h]), pct(at[h][100]["resampled_finite_sensitivity"])]
    figures.append(pct(flags_all_among_defined(at["control"][100])))
    figures += [str(min(below_half.values())), str(max(below_half.values()))]
    assert not missing_from(refitted, figures), missing_from(refitted, figures)
    certified = ppv_md_part("### 3.3", "### 3.4")
    figures = [pct1(columbia[0]), pct1(columbia[-1])]
    figures += [pct1(at[h][100]["certified_violation_rate_given_threshold"]) for h in MAIN]
    assert not missing_from(certified, figures), missing_from(certified, figures)
    nine_sorted = sorted(nine.values())
    assert f"{nine_sorted[0]:.0f} to {nine_sorted[-1]:.0f} labels" in ppv_md_section()


def test_ppv_md_post_hoc_claims_are_the_file_s(result: dict[str, Any]) -> None:
    """The prose around Tables 2 and 3: case counts, flag-everyone shares, net benefit gaps."""
    s = result["summary"]
    at = {h: next(b for b in s[h]["budgets"] if b["n"] == TABLE_BUDGET) for h in SITE_NAMES}
    few = {h for h in SITE_NAMES if at[h]["mean_cases_labelled"] < 2}
    assert few == {"shandong", "code15", "zzu"}
    assert max(at[h]["mean_cases_labelled"] for h in SITE_NAMES) < 16
    shares = [flags_all_among_defined(at[h]) for h in SITE_NAMES]
    always = {h for h in SITE_NAMES if flags_all_among_defined(at[h]) == pytest.approx(1.0)}
    assert always == {"shandong", "code15", "code_test", "zzu"}
    gap = {
        h: at[h]["net_benefit"]["resampled"] - s[h]["references"]["treat_all_net_benefit"]
        for h in SITE_NAMES
    }
    assert {h for h, g in gap.items() if abs(g) > 0.006} == {"ptbxl"}
    assert gap["ptbxl"] < 0
    assert all(
        s[h]["prior"]["net_benefit"] > s[h]["references"]["treat_all_net_benefit"]
        for h in SITE_NAMES
    )
    reserve = [
        s[h]["prior"]["abs_error"]["median"] for h in SITE_NAMES if h not in (*MAIN, "control")
    ]
    figures = [
        pct(min(shares)),
        pct(max(shares)),
        f"{abs(gap['ptbxl']):.3f}",
        signed(at["columbia"]["net_benefit"]["resampled"]),
        signed(s["columbia"]["references"]["treat_all_net_benefit"]),
        signed(s["columbia"]["prior"]["net_benefit"]),
    ]
    reserve_prose = ppv_md_part("### 3.5", "### 3.6")
    assert not missing_from(reserve_prose, figures), missing_from(reserve_prose, figures)
    label_free = ppv_md_part("### 3.1", "### 3.2")
    assert f"{pts(min(reserve))} to {pts(max(reserve))} points" in label_free


def english_form(text: str) -> str:
    """French number forms back to English: 13,1 to 13.1, 90 % to 90%."""
    text = re.sub(r"(\d),(\d)", r"\1.\2", text)
    return text.replace(" %", "%")


def test_ppv_fr_md_mirrors_ppv_md() -> None:
    """The French version has the English plan, table for table and figure for figure."""
    en = (REPO_ROOT / "PPV.md").read_text()
    fr = (REPO_ROOT / "PPV.fr.md").read_text()

    def plan(text: str) -> list[str]:
        return [re.sub(r"\D", "", h) for h in re.findall(r"(?m)^#{2,3} .*$", text)]

    def rows(text: str) -> list[list[str]]:
        lines = [x for x in text.splitlines() if x.startswith("|") and not x.startswith("|-")]
        return [re.findall(r"−?\d+(?:\.\d+)?%?", english_form(x)) for x in lines]

    assert plan(fr) == plan(en)
    assert rows(fr) == rows(en)
    figures = r"−?\d+(?:\.\d+)?%?"
    assert sorted(re.findall(figures, english_form(fr))) == sorted(re.findall(figures, en))

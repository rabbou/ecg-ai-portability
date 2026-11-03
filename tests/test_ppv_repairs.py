"""The five PPV repairs of protocol section 7, on values worked out by hand."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest
from ppv_repairs import freeze_targets, key_of, pooled_error_bounds
from score_pairs import Pair

from ecs.ppv_repairs import (
    Target,
    certified_cut,
    clopper_pearson_lower,
    frozen_target,
    labelled_draw,
    prior_repair,
    references,
    sensitivity_cut,
    transfer_cell,
    weighted_conformal_selection,
)


def target_with(grid: tuple[float, ...], v: float) -> Target:
    return Target(cut=0.5, ppv=v, sensitivity=0.9, specificity=0.5, grid=grid)


class TestFrozenTarget:
    def test_nine_ill_flagged_and_one_healthy_gives_point_nine(self) -> None:
        """A wrong v puts every repair's yardstick in the wrong place.

        Nine ill at p = 0.95 .. 0.55 have scores 0.05 .. 0.45; the conformal rank
        ceil(10 * 0.9) = 9 makes the cut 0.45.  Of the healthy (0.6, 0.5, 0.2, 0.1),
        only 0.6 scores under it, so the source flags 10 and v = 9 / 10.
        """
        ill = np.round(np.arange(0.95, 0.54, -0.05), 2)
        p = np.concatenate([ill, [0.6, 0.5, 0.2, 0.1]])
        y = np.array([1] * 9 + [0] * 4)
        target = frozen_target(p, y)
        assert target.cut == pytest.approx(0.45)
        assert target.ppv == pytest.approx(0.9)
        assert target.sensitivity == pytest.approx(1.0)
        assert target.specificity == pytest.approx(0.75)

    def test_freezing_reads_no_target_record(self) -> None:
        """Protocol section 7: no target v may depend on an evaluation score.

        The same pairs with their target arrays replaced by NaN freeze to the
        same table, so nothing of the target enters it.
        """
        rng = np.random.default_rng(3)
        pairs = []
        for k in range(3):
            y_s = (rng.random(400) < 0.3).astype(int)
            p_s = np.clip(0.3 + 0.4 * y_s + rng.normal(0, 0.2, 400), 0.01, 0.99)
            y_t = (rng.random(300) < 0.1).astype(int)
            p_t = rng.random(300)
            pairs.append(Pair("f", "m", "s", f"t{k}", "L", p_s, y_s, p_t, y_t, None))
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
            for p in pairs
        ]
        assert freeze_targets(pairs) == freeze_targets(blind)
        assert len(freeze_targets(pairs)) == 3
        assert key_of(pairs[0]) == "f/m/s/t0/L"


class TestClopperPearson:
    @pytest.mark.parametrize(
        ("ill", "flagged", "bound"),
        [(3, 3, 0.1 ** (1 / 3)), (1, 1, 0.1), (0, 7, 0.0)],
    )
    def test_closed_forms(self, ill: int, flagged: int, bound: float) -> None:
        """x = k gives delta^(1/k); x = 0 gives 0: the bound the certificate rests on."""
        assert clopper_pearson_lower(ill, flagged) == pytest.approx(bound, rel=1e-9)


def labelled(cuts: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """Records whose score 1 - p sits in bands below 0.1, 0.3, 0.5 and 0.6: (ill, healthy)."""
    p, y = [], []
    for score, (ill, healthy) in zip((0.05, 0.2, 0.4, 0.55), cuts, strict=False):
        p += [1 - score] * (ill + healthy)
        y += [1] * ill + [0] * healthy
    return np.array(p), np.array(y)


class TestCertifiedCut:
    def test_the_walk_stops_at_the_first_null_it_keeps(self) -> None:
        """A certificate past a failed test would break the fixed-sequence guarantee.

        At v = 0.5: cut 0.1 flags 5 ill, P(X >= 5) = 1/32 = 0.031; cut 0.3 flags
        8 ill of 10, P(X >= 8) = 56/1024 = 0.055; cut 0.5 flags 12 of 20,
        P(X >= 12) = 0.252 > 0.1.  The certified cut is 0.3, although cut 0.6, which
        adds 20 ill (32 of 40, P(X >= 32) < 0.001), would pass if the walk went on.
        """
        p, y = labelled([(5, 0), (3, 2), (4, 6), (20, 0)])
        got = certified_cut(p, y, target_with((0.1, 0.3, 0.5, 0.6), 0.5))
        assert got == pytest.approx(0.3)

    def test_an_empty_top_threshold_certifies_nothing(self) -> None:
        """The twin: a top cut flagging no one stops the walk, whatever lies below."""
        p, y = labelled([(5, 0), (3, 2), (4, 6)])
        assert certified_cut(p, y, target_with((0.01, 0.1, 0.3), 0.5)) is None

    def test_v_above_the_evidence_certifies_nothing(self) -> None:
        """At v = 0.6 the top cut's P(X >= 5 | 5, 0.6) = 0.078 passes and cut 0.3's
        P(X >= 8 | 10, 0.6) = 0.167 fails, so only cut 0.1 is certified."""
        p, y = labelled([(5, 0), (3, 2), (4, 6)])
        assert certified_cut(p, y, target_with((0.1, 0.3, 0.5), 0.6)) == pytest.approx(0.1)


class TestLabelledDraw:
    @pytest.mark.parametrize("seed", range(6))
    def test_no_evaluation_patient_has_a_labelled_record(self, seed: int) -> None:
        """A patient on both sides would let the labels see the evaluation."""
        codes = np.array([0, 0, 1, 1, 1, 2, 3, 3, 4, 5])
        lab, ev = labelled_draw(codes, 3, np.random.default_rng(seed))
        assert len(lab) == 3
        assert not set(codes[lab]) & set(codes[ev])
        for patient in set(codes[ev]):
            assert (codes[ev] == patient).sum() == (codes == patient).sum()

    def test_one_record_per_patient_leaves_the_rest_to_evaluation(self) -> None:
        codes = np.arange(10)
        lab, ev = labelled_draw(codes, 4, np.random.default_rng(0))
        assert sorted([*lab, *ev]) == list(range(10))
        assert len(ev) == 6


class TestWeightedConformalSelection:
    CAL_P = np.array([0.1, 0.2, 0.3, 0.9])
    CAL_Y = np.array([0, 0, 0, 1])
    TEST_P = np.array([0.95, 0.85, 0.15])

    def test_hand_case_selects_the_two_high_scores(self) -> None:
        """Jin & Candes' Algorithm 1 worked by hand, unit weights, q = 0.5.

        p-values (A + 1) / 5: 0.2, 0.2, 0.6.  For j = 0.95 the auxiliary values are
        (0, 0.2, 0.6), BH size 2, so 0.2 <= 0.5 * 2 / 3 passes; j = 0.85 gives
        (0, 0, 0.6), size 2, passes; j = 0.15 gives (0, 0, 0), size 3, and
        0.6 > 0.5 fails.  Pruning: two candidates of size 2, r* = 2.
        """
        ones = np.ones(4)
        got = weighted_conformal_selection(
            self.CAL_P, self.CAL_Y, ones, self.TEST_P, np.ones(3), 0.5
        )
        assert got.tolist() == [True, True, False]

    def test_a_stricter_level_selects_nothing(self) -> None:
        """The twin at q = 0.2: sizes 1 and 2 give cut-offs 0.067 and 0.133, under 0.2."""
        got = weighted_conformal_selection(
            self.CAL_P, self.CAL_Y, np.ones(4), self.TEST_P, np.ones(3), 0.2
        )
        assert not got.any()

    @pytest.mark.parametrize("seed", range(4))
    def test_agrees_with_the_algorithm_written_out(self, seed: int) -> None:
        """The vectorised positions against Algorithm 1 run loop by loop, unequal weights."""
        rng = np.random.default_rng(seed)
        cal_y = (rng.random(40) < 0.4).astype(int)
        cal_p = np.clip(0.35 + 0.3 * cal_y + rng.normal(0, 0.2, 40), 0.01, 0.99)
        test_p = rng.random(25)
        w_cal, w = rng.uniform(0.5, 2.0, 40), rng.uniform(0.5, 2.0, 25)
        got = weighted_conformal_selection(cal_p, cal_y, w_cal, test_p, w, 0.3)
        assert got.tolist() == slow_wcs(cal_p, cal_y, w_cal, test_p, w, 0.3)

    @pytest.mark.parametrize("seed", range(4))
    def test_tied_scores_follow_the_strict_inequalities(self, seed: int) -> None:
        """Scores on a 0.1 grid tie often; a tie must not count as scored above."""
        rng = np.random.default_rng(seed)
        cal_y = (rng.random(40) < 0.4).astype(int)
        cal_p = np.round(np.clip(0.35 + 0.3 * cal_y + rng.normal(0, 0.2, 40), 0.05, 0.95), 1)
        test_p = np.round(rng.uniform(0.05, 0.95, 25), 1)
        w_cal, w = rng.uniform(0.5, 2.0, 40), rng.uniform(0.5, 2.0, 25)
        got = weighted_conformal_selection(cal_p, cal_y, w_cal, test_p, w, 0.4)
        assert got.tolist() == slow_wcs(cal_p, cal_y, w_cal, test_p, w, 0.4)


def bh_size(values: list[float], q: float) -> int:
    ordered = sorted(values)
    m = len(ordered)
    passing = [k for k in range(1, m + 1) if ordered[k - 1] <= q * k / m]
    return max(passing) if passing else 0


def slow_wcs(
    cal_p: np.ndarray,
    cal_y: np.ndarray,
    w_cal: np.ndarray,
    test_p: np.ndarray,
    w: np.ndarray,
    q: float,
) -> list[bool]:
    """Algorithm 1 with the clipped score, deterministic pruning, one loop per definition."""
    m, total = len(test_p), float(sum(w_cal))

    def below(ell: int) -> float:
        return float(
            sum(
                wi
                for pi, yi, wi in zip(cal_p, cal_y, w_cal, strict=True)
                if yi == 0 and pi > test_p[ell]
            )
        )

    pvals = [(below(ell) + w[ell]) / (total + w[ell]) for ell in range(m)]
    sizes = []
    for j in range(m):
        aux = [
            0.0 if ell == j else (below(ell) + w[j] * (test_p[j] > test_p[ell])) / (total + w[j])
            for ell in range(m)
        ]
        sizes.append(bh_size(aux, q))
    first = [pvals[j] <= q * sizes[j] / m for j in range(m)]
    r_star = max(
        [
            r
            for r in range(1, m + 1)
            if sum(f and s <= r for f, s in zip(first, sizes, strict=True)) >= r
        ],
        default=0,
    )
    return [f and s <= r_star for f, s in zip(first, sizes, strict=True)]


def gaussian(
    n: int, prior: float, shift: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Scores calibrated at prevalence 0.5 for x ~ N(+-1, 1); the ill moved by ``shift``."""
    y = (rng.random(n) < prior).astype(int)
    x = rng.normal(np.where(y == 1, 1.0 + shift, -1.0), 1.0)
    return 1.0 / (1.0 + np.exp(-2.0 * x)), y


class TestPriorRepair:
    def test_a_pure_prevalence_shift_is_announced_within_two_points(self) -> None:
        """Label shift alone: EM's prevalence and Bayes' rule give the observed PPV."""
        rng = np.random.default_rng(0)
        p_s, y_s = gaussian(4000, 0.5, 0.0, rng)
        p_t, y_t = gaussian(20000, 0.15, 0.0, rng)
        out = prior_repair(p_s, y_s, p_t, y_t, frozen_target(p_s, y_s))
        assert out["abs_error"] < 0.02

    def test_a_moved_disease_is_not_announced_within_five_points(self) -> None:
        """The twin: the ill ECGs look healthier at the target, which no prior fixes."""
        rng = np.random.default_rng(0)
        p_s, y_s = gaussian(4000, 0.5, 0.0, rng)
        p_t, y_t = gaussian(20000, 0.15, -1.2, rng)
        out = prior_repair(p_s, y_s, p_t, y_t, frozen_target(p_s, y_s))
        assert out["abs_error"] > 0.05


class TestTransferCell:
    def test_selection_runs_at_one_minus_the_frozen_v(self) -> None:
        """Method 5's level is q = 1 - v of the frozen target, not a value read at the target."""
        rng = np.random.default_rng(1)
        p_s, y_s = gaussian(1000, 0.3, 0.0, rng)
        p_t, y_t = gaussian(1200, 0.3, 0.0, rng)
        target = Target(cut=0.4, ppv=0.8, sensitivity=0.9, specificity=0.9, grid=(0.1, 0.2))
        cell = transfer_cell(p_s, y_s, p_t, y_t, pd.Series(range(1200)), target, 0, (100,))
        assert cell["selection"]["q"] == pytest.approx(0.2)

    def test_certification_without_shift_violates_at_most_delta_plus_noise(self) -> None:
        """Same population on both sides: the certificate's violation rate stays near delta.

        At the source PPV v, a lower threshold certified by chance would read a
        PPV under v; the fixed-sequence test bounds that at delta = 0.1, and the
        evaluation half's own sampling noise adds a few points.
        """
        rng = np.random.default_rng(2)
        p_s, y_s = gaussian(3000, 0.3, 0.0, rng)
        p_t, y_t = gaussian(3000, 0.3, 0.0, rng)
        target = frozen_target(p_s, y_s)
        cell = transfer_cell(p_s, y_s, p_t, y_t, pd.Series(range(3000)), target, 0, (200,))
        rung = cell["budgets"][0]
        assert rung["certified"]["share_threshold_exists"] > 0.5
        assert rung["certified"]["violation_rate"] <= 0.15

    def test_budgets_over_half_the_target_are_not_run(self) -> None:
        """n labels must leave an evaluation set at least as large."""
        rng = np.random.default_rng(4)
        p_s, y_s = gaussian(600, 0.3, 0.0, rng)
        p_t, y_t = gaussian(300, 0.3, 0.0, rng)
        cell = transfer_cell(
            p_s, y_s, p_t, y_t, pd.Series(range(300)), frozen_target(p_s, y_s), 0, (100, 200)
        )
        assert [b["n"] for b in cell["budgets"]] == [100]
        assert math.isfinite(cell["prior"]["announced"])

    def test_nine_cases_is_the_least_that_places_a_threshold(self) -> None:
        """ceil((m + 1) * 0.9) exceeds m below nine cases: the cut is +inf."""
        for m, bounded in ((8, False), (9, True)):
            p = np.linspace(0.5, 0.99, m)
            cut = sensitivity_cut(p, np.ones(m, dtype=int))
            assert cut is not None and math.isfinite(cut) is bounded

    def test_a_half_under_nine_cases_flags_every_patient(self) -> None:
        """At 3% prevalence 100 labels hold about three cases, too few for the conformal rank."""
        rng = np.random.default_rng(5)
        p_s, y_s = gaussian(3000, 0.3, 0.0, rng)
        p_t, y_t = gaussian(4000, 0.03, 0.0, rng)
        cell = transfer_cell(
            p_s, y_s, p_t, y_t, pd.Series(range(4000)), frozen_target(p_s, y_s), 0, (100,)
        )
        rung = cell["budgets"][0]
        assert rung["mean_cases_labelled"] < 9
        # The other draws hold no case at all in the first half and return no cut.
        assert rung["resampled"]["share_flags_all"] > 0.75
        # Every draw that placed a threshold flagged everyone; a draw with no case has none.
        assert rung["resampled"]["share_flags_all"] == pytest.approx(
            rung["resampled"]["share_defined"]
        )
        assert rung["resampled"]["sensitivity"] == pytest.approx(1.0)
        assert rung["resampled"]["share_finite"] == 0.0
        assert rung["resampled"]["finite_abs_error"] is None
        assert rung["resampled"]["finite_sensitivity"] is None
        assert cell["labels_for_nine_cases_per_half"] == pytest.approx(18 / y_t.mean())

    def test_enough_cases_place_a_finite_threshold(self) -> None:
        """The twin: at 30% prevalence 400 labels hold about 60 cases per half."""
        rng = np.random.default_rng(6)
        p_s, y_s = gaussian(3000, 0.3, 0.0, rng)
        p_t, y_t = gaussian(4000, 0.3, 0.0, rng)
        cell = transfer_cell(
            p_s, y_s, p_t, y_t, pd.Series(range(4000)), frozen_target(p_s, y_s), 0, (400,)
        )
        rung = cell["budgets"][0]
        assert rung["mean_cases_labelled"] == pytest.approx(120, rel=0.15)
        assert rung["resampled"]["share_flags_all"] == 0.0
        assert rung["resampled"]["share_finite"] == pytest.approx(1.0)
        assert rung["resampled"]["finite_abs_error"] == rung["resampled"]["abs_error"]
        assert rung["resampled"]["finite_sensitivity"] < 0.97

    def test_finite_draws_are_kept_apart_when_both_kinds_occur(self) -> None:
        """At 10% prevalence 200 labels hold about ten cases per half: some cuts are +inf."""
        rng = np.random.default_rng(11)
        p_s, y_s = gaussian(3000, 0.3, 0.0, rng)
        p_t, y_t = gaussian(4000, 0.1, 0.0, rng)
        cell = transfer_cell(
            p_s, y_s, p_t, y_t, pd.Series(range(4000)), frozen_target(p_s, y_s), 0, (200,)
        )
        out = cell["budgets"][0]["resampled"]
        assert 0.2 < out["share_finite"] < 0.8
        assert out["finite_sensitivity"] < out["sensitivity"]
        assert out["finite_abs_error"] != out["abs_error"]


class TestReferences:
    def test_treat_all_is_prevalence_minus_weighted_non_cases(self) -> None:
        y = np.array([1] * 20 + [0] * 80)
        p = np.linspace(0.01, 0.99, 100)
        out = references(p, y)
        assert out["treat_all_net_benefit"] == pytest.approx(0.2 - 0.8 * 0.1 / 0.9)

    def test_the_target_own_cut_beats_treat_all_on_a_separating_score(self) -> None:
        rng = np.random.default_rng(7)
        p, y = gaussian(4000, 0.1, 0.0, rng)
        out = references(p, y)
        assert out["own_cut_sensitivity"] >= 0.9
        assert out["own_cut_net_benefit"] > out["treat_all_net_benefit"]


def rung_of(errors: list[list[float]]) -> list[dict[str, Any]]:
    """Budget rungs whose stated PPV errs by each given amount against an observed 0.5."""
    return [
        {"_pairs": {"recalibrated": [(0.5 + e, 0.5) for e in draws], "resampled": []}}
        for draws in errors
    ]


def bounds_of(rungs: list[dict[str, Any]]) -> tuple[float, float]:
    out = pooled_error_bounds(rungs, "recalibrated", 0)
    assert out is not None
    return out


class TestPooledErrorBounds:
    def test_the_interval_brackets_the_pooled_median(self) -> None:
        rng = np.random.default_rng(8)
        rungs = rung_of([list(rng.uniform(0, 0.1, 50)) for _ in range(10)])
        pooled = np.median(
            [e for r in rungs for a, o in r["_pairs"]["recalibrated"] for e in [a - o]]
        )
        low, high = bounds_of(rungs)
        assert low <= pooled <= high

    def test_spread_between_transfers_widens_the_interval(self) -> None:
        """Transfers are redrawn first: medians far apart give a wider interval."""
        rng = np.random.default_rng(9)
        alike = rung_of([list(rng.uniform(0.04, 0.06, 50)) for _ in range(10)])
        apart = rung_of([list(rng.uniform(0.01 * i, 0.01 * i + 0.002, 50)) for i in range(10)])
        a_low, a_high = bounds_of(alike)
        b_low, b_high = bounds_of(apart)
        assert b_high - b_low > 3 * (a_high - a_low)

    def test_one_transfer_is_redrawn_on_its_draws(self) -> None:
        rng = np.random.default_rng(10)
        rungs = rung_of([list(rng.uniform(0, 0.1, 200))])
        low, high = bounds_of(rungs)
        assert 0 < high - low < 0.03

    def test_a_method_with_no_draw_has_no_interval(self) -> None:
        assert pooled_error_bounds(rung_of([[0.01]]), "resampled", 0) is None

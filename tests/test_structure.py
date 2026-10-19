"""Study P-A's measurements: the weighted per-label repair, its verdict, and the A3 metrics.

Each test names the result in STRUCTURE.md that a broken function would make wrong.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ecs.calibration import integrated_calibration_index
from ecs.structure import (
    a2_verdict,
    augrc,
    calibration_cell,
    domain_weights,
    effective_cases,
    holm_rejections,
    selective_sensitivity,
    superiority_p,
    weighted_perlabel_sets,
)
from ecs.transfer import conformal_sets


def _expit(z: np.ndarray) -> np.ndarray:
    return np.asarray(1 / (1 + np.exp(-z)))


def _shifted_sample(
    rng: np.random.Generator, n: int, mean: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """x ~ N(mean, 1), P(y=1|x) = expit(2x - 1) in every population, p the true risk."""
    x = rng.normal(mean, 1.0, n)
    p = _expit(2 * x - 1)
    return x, p, (rng.random(n) < p).astype(int)


class TestWeightedPerLabel:
    def test_unit_weights_give_the_unweighted_per_label_sets(self) -> None:
        """With w = 1 everywhere the repair must be the study's per-label method, or the
        weighted row of structure_coverage.csv is not comparable with the published one."""
        rng = np.random.default_rng(0)
        _, p_cal, y_cal = _shifted_sample(rng, 800, 0.0)
        _, p_tgt, _ = _shifted_sample(rng, 300, 0.0)
        weighted = weighted_perlabel_sets(
            p_cal, y_cal, np.ones(800), p_tgt, np.ones(300), alpha=0.1
        )
        np.testing.assert_array_equal(weighted, conformal_sets(p_cal, y_cal, p_tgt)["perlabel"])

    def test_true_weights_restore_the_sensitivity_a_covariate_shift_removes(self) -> None:
        """Under a pure covariate shift the weighted threshold must give back the 90% the
        unweighted one loses, or falsifier A2 judges a broken repair.

        Source x ~ N(0, 1), target x ~ N(-1, 1): target cases sit lower on the risk
        scale, so the source threshold misses more of them.  The true ratio is
        N(-1,1)/N(0,1) = exp(-x - 1/2).
        """
        rng = np.random.default_rng(1)
        x_cal, p_cal, y_cal = _shifted_sample(rng, 4000, 0.0)
        x_tgt, p_tgt, y_tgt = _shifted_sample(rng, 3000, -1.0)
        plain = conformal_sets(p_cal, y_cal, p_tgt)["perlabel"]
        weighted = weighted_perlabel_sets(
            p_cal, y_cal, np.exp(-x_cal - 0.5), p_tgt, np.exp(-x_tgt - 0.5), alpha=0.1
        )
        cases = y_tgt == 1
        assert plain[cases, 1].mean() < 0.86
        assert 0.86 <= weighted[cases, 1].mean() <= 0.94


class TestDomainWeights:
    def test_two_draws_of_one_population_weigh_about_one(self) -> None:
        """Weights fitted between identical populations must stay near 1, or the
        inpatient-to-inpatient row would show a repair where nothing moved."""
        rng = np.random.default_rng(2)
        w = domain_weights(rng.normal(size=(300, 3)), rng.normal(size=(200, 3)))
        assert w.domain_auroc < 0.58
        assert 0.9 < w.calibration.mean() < 1.1

    def test_a_shifted_population_recovers_the_true_density_ratio(self) -> None:
        """The fitted w(x) must track p_t(x)/p_s(x), the quantity the guarantee needs."""
        rng = np.random.default_rng(3)
        x_cal, x_tgt = rng.normal(0, 1, (600, 1)), rng.normal(-1, 1, (400, 1))
        w = domain_weights(x_cal, x_tgt)
        truth = np.exp(-x_cal[:, 0] - 0.5)
        kept = truth <= w.cap  # the cap clips the tail on purpose
        assert w.domain_auroc > 0.70
        assert np.corrcoef(w.calibration[kept], truth[kept])[0, 1] > 0.95
        assert w.calibration.max() == pytest.approx(w.cap)


class TestVerdict:
    def test_holm_steps_down_and_stops_at_the_first_failure(self) -> None:
        """Holm on the judged anomalies: a wrong step-down passes anomalies the protocol fails."""
        assert holm_rejections({"a": 0.01, "b": 0.02, "c": 0.03}, 0.05) == {
            "a": True,
            "b": True,
            "c": True,
        }
        # 0.03 > 0.05 / 2 stops the procedure; 0.04 is never reached.
        assert holm_rejections({"a": 0.01, "b": 0.04, "c": 0.03}, 0.05) == {
            "a": True,
            "b": False,
            "c": False,
        }

    def test_the_score_test_p_value_by_hand(self) -> None:
        """z = (0.9 - 0.8) / sqrt(0.16 / 100) = 2.5, one-sided p = 0.00621."""
        assert superiority_p(0.9, 100) == pytest.approx(0.0062097, abs=1e-6)
        assert superiority_p(0.9, 0) == 1.0

    def test_effective_cases_is_the_smaller_sample(self) -> None:
        """The Kish size of 4 unit weights is 4; of (1, 1, 1, 9) it is 144 / 84 = 1.71."""
        assert effective_cases(10, np.ones(4)) == 4
        assert effective_cases(3, np.ones(40)) == 3
        assert effective_cases(10, np.array([1.0, 1.0, 1.0, 9.0])) == 1

    @staticmethod
    def _row(sensitivity: float, n: int) -> dict[str, float]:
        return {"coverage_pos": sensitivity, "n_effective": n}

    def test_the_composite_band_is_judged_on_the_effective_size(self) -> None:
        """90% on 400 cases is inside [85%, 95%] (Wilson 86.7-92.6%); on 100 it is not
        (Wilson 82.6-94.5%), so the same sensitivity passes or fails by the weights' size."""
        strong = {"x": self._row(0.95, 400)}
        assert a2_verdict(self._row(0.9, 400), strong)["composite_pass"] is True
        assert a2_verdict(self._row(0.9, 100), strong)["composite_pass"] is False
        # 97% on 400 (Wilson 94.8-98.3%) overshoots the band: a threshold that flags far
        # more than 90% of cases is not the 90% the inpatient threshold promised.
        assert a2_verdict(self._row(0.97, 400), strong)["composite_pass"] is False

    def test_the_repair_needs_a_strict_majority_of_judged_anomalies(self) -> None:
        """Three of four anomalies above 80% hold the repair; two of four do not."""
        good, bad = self._row(0.95, 400), self._row(0.70, 400)
        composite = self._row(0.9, 400)
        three = {"a": good, "b": good, "c": good, "d": bad}
        two = {"a": good, "b": good, "c": bad, "d": bad}
        assert a2_verdict(composite, three)["repair_holds"] is True
        assert a2_verdict(composite, two)["repair_holds"] is False
        assert a2_verdict(self._row(0.8, 400), three)["repair_holds"] is False


class TestSelective:
    # y, p and the threshold 0.5: flags (1, 1, 0, 0), wrong at records 1 and 2.
    # |logit p| is 2.197, 1.386, 1.386, 2.197, so the order kept is 0, 3, 1, 2.
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.8, 0.2, 0.1])

    def test_augrc_by_hand(self) -> None:
        """Cumulative wrong 0, 0, 1, 2 over n = 4: AUGRC = (0 + 0 + 0.25 + 0.5) / 4."""
        assert augrc(self.y, self.p, 0.5) == pytest.approx(0.1875)

    def test_sensitivity_on_the_kept_half(self) -> None:
        """Keeping records 0 and 3: the one case kept is flagged, half the cases remain."""
        assert selective_sensitivity(self.y, self.p, 0.5, retained=0.5) == (1.0, 0.5)
        assert selective_sensitivity(np.zeros(4, dtype=int), self.p, 0.5) == (None, 0.0)


class TestCalibration:
    def test_ici_is_small_when_calibrated_and_large_when_miscalibrated(self) -> None:
        """The ICI column must separate a calibrated model from one 0.2 too high, and
        from an overconfident one whose errors cancel on average (mean |gap| 0.15)."""
        rng = np.random.default_rng(4)
        p = rng.uniform(0.1, 0.7, 20000)
        y = (rng.random(20000) < p).astype(int)
        assert integrated_calibration_index(y, p) < 0.02
        assert integrated_calibration_index(y, p + 0.2) > 0.17
        overconfident = 0.4 + 2 * (p - 0.4)  # gap |p' - p| = |p - 0.4|, mean 0.15
        assert integrated_calibration_index(y, np.clip(overconfident, 0, 1)) > 0.12

    def test_the_cell_brackets_each_point_and_counts_its_draws(self) -> None:
        """Every interval of structure_calibration.json holds its point, and a metric
        undefined on every draw is reported as absent, not as zero."""
        rng = np.random.default_rng(5)
        p = rng.uniform(0.05, 0.95, 400)
        y = (rng.random(400) < p).astype(int)
        cell = calibration_cell(y, p, threshold=0.5, draws=20)
        for key in ("slope", "ici", "brier", "augrc", "sensitivity_at_80"):
            assert cell[f"{key}_low"] <= cell[key] <= cell[f"{key}_high"]
        assert cell["brier_draws"] == 20
        empty = calibration_cell(np.zeros(50, dtype=int), p[:50], threshold=0.5, draws=10)
        assert empty["sensitivity_at_80"] is None
        assert empty["sensitivity_at_80_low"] is None
        assert empty["slope_draws"] == 0
        assert math.isclose(empty["brier"], float(np.mean(p[:50] ** 2)))

"""MIMIC-IV-ECG cart timeline and shadow cohorts, on synthetic releases.

The open shadow run places tracings in time from the carts alone; these tests
hold the least-squares chronology and the cohort draw to hand-built releases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ecs.mimic import (
    cart_chronology,
    estimated_eras,
    estimated_years,
    shadow_cohorts,
)


def _synthetic_release(seed: int = 0) -> tuple[pd.DataFrame, dict[int, float]]:
    """Twelve carts at known times, patients shifted by large random offsets."""
    rng = np.random.default_rng(seed)
    carts = {100 + c: 2008.0 + c for c in range(12)}
    rows = []
    study = 0
    for patient in range(3000):
        offset = rng.uniform(100, 200)
        for _ in range(rng.integers(1, 5)):
            cart = int(rng.integers(100, 112))
            real = carts[cart] + rng.normal(0, 0.3)
            shifted = real + offset
            year = int(shifted)
            day = int((shifted - year) * 365) + 1
            stamp = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=day - 1)
            rows.append(
                {
                    "study_id": study,
                    "subject_id": patient,
                    "cart_id": cart,
                    "ecg_time": stamp.strftime("%Y-%m-%d"),
                    "real": real,
                }
            )
            study += 1
    return pd.DataFrame(rows), carts


class TestTheCartTimeline:
    def test_carts_are_placed_where_they_were_in_time(self) -> None:
        frame, carts = _synthetic_release()
        chronology = cart_chronology(frame)
        truth = pd.Series(carts) - float(np.median(list(carts.values())))
        recovered = chronology.position.loc[truth.index]
        assert np.abs(recovered - truth).max() < 0.15

    def test_a_patient_whose_tracings_span_decades_is_left_out(self) -> None:
        frame, _ = _synthetic_release()
        clean = cart_chronology(frame).position
        broken = frame.copy()
        broken.loc[broken["subject_id"] == 0, "ecg_time"] = "2299-01-01"
        broken.loc[broken.index[0], "ecg_time"] = "2101-01-01"
        assert np.allclose(cart_chronology(broken).position, clean, atol=0.05)

    def test_the_estimate_orders_tracings_by_their_real_time(self) -> None:
        frame, _ = _synthetic_release()
        estimate = estimated_years(frame, cart_chronology(frame))
        real = frame.set_index(frame["study_id"].astype(str))["real"]
        rho = pd.Series(estimate).corr(real.loc[estimate.index], method="spearman")
        assert rho > 0.95

    def test_eras_are_terciles(self) -> None:
        era = estimated_eras(pd.Series(np.arange(9.0), index=list("abcdefghi")))
        assert era.tolist() == ["early"] * 3 + ["middle"] * 3 + ["late"] * 3


class TestTheShadowCohorts:
    def test_cohorts_are_disjoint_eligible_and_of_the_stated_sizes(self) -> None:
        ids = [str(i) for i in range(30000)]
        era = pd.Series(["early", "late", "middle"] * 10000, index=ids)
        eligible = pd.Index(ids[:27000])
        cohorts = shadow_cohorts(eligible, era)
        assert {k: len(v) for k, v in cohorts.items()} == {
            "early_cal": 2000,
            "early_test": 4000,
            "late_test": 8000,
        }
        seen: set[str] = set()
        for name, members in cohorts.items():
            assert not seen & set(members), name
            seen |= set(members)
            assert set(members) <= set(eligible)
        assert set(era[cohorts["late_test"]]) == {"late"}
        assert set(era[cohorts["early_cal"] + cohorts["early_test"]]) == {"early"}
        assert cohorts == shadow_cohorts(eligible, era)

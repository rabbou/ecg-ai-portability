"""MIMIC-IV-ECG statement rules and the one-tracing-per-patient draw.

The rule tests are an oracle written by hand from the statements as the carts
print them, not from the patterns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ecs.mimic import (
    AMBIGUITIES,
    CLASSES,
    POSITIVE_STATUSES,
    RULES,
    STATUSES,
    classify,
    normalise,
    one_per_patient,
    real_eras,
    real_year_bounds,
    record_labels,
    record_statuses,
)

# Statement as printed by a cart -> the status it must get, per class.
ORACLE: dict[str, dict[str, str]] = {
    "Sinus rhythm": {"NSR": "definite"},
    "SINUS RHYTHM": {"NSR": "definite"},
    "Sinus rhythm with borderline 1st degree A-V block.": {
        "NSR": "definite",
        "IAVB": "borderline",
    },
    "Sinus bradycardia with 1st degree A-V block": {"NSR": "excluded", "IAVB": "definite"},
    "Sinus tachycardia": {"NSR": "excluded"},
    "Sinus or ectopic atrial rhythm": {"NSR": "excluded"},
    "Sinus bradycardia - consider sinus rhythm with 2:1 S-A block.": {"NSR": "hedged"},
    "Atrial fibrillation with rapid ventricular response": {"AF": "definite"},
    "ATRIAL FIBRILLATION": {"AF": "definite"},
    "Probable atrial fibrillation with PVC(s)": {"AF": "hedged"},
    "Atrial flutter/fibrillation": {"AF": "excluded"},
    "Afib/flutter and ventricular-paced rhythm": {"AF": "excluded"},
    "Atrial flutter with 4:1 A-V block": {},
    "Left bundle branch block": {"LBBB": "definite"},
    "Incomplete LBBB": {"LBBB": "excluded"},
    "Conduction defect of LBBB type": {"LBBB": "excluded"},
    "Anterior Q waves, possibly due to ILBBB": {"LBBB": "excluded"},
    "RBBB with left anterior fascicular block": {"RBBB": "definite"},
    "RBBB with RAD - possible left posterior fascicular block": {"RBBB": "definite"},
    "Incomplete right bundle branch block": {"RBBB": "excluded"},
    "IVCD, consider atypical RBBB": {"RBBB": "excluded"},
    "I.V. CONDUCTION DEFECT OF RIGHT BUNDLE BRANCH BLOCK TYPE": {"RBBB": "excluded"},
    "Prolonged PR interval": {"IAVB": "definite"},
    "Borderline prolonged PR interval": {"IAVB": "borderline"},
    "- first degree A-V block": {"IAVB": "definite"},
    "Short PR interval": {},
    "Second degree AV block, Mobitz II": {},
    "Inferior infarct, old": {"MI_OLD": "definite"},
    "Old inferior infarct": {"MI_OLD": "definite"},
    "Inferior infarct - age undetermined": {"MI_OLD": "age_undetermined"},
    "Anteroseptal infarct, age indeterminate": {"MI_OLD": "age_undetermined"},
    "Possible anterior infarct - age undetermined": {"MI_OLD": "hedged"},
    "Possible old inferior infarct": {"MI_OLD": "hedged"},
    "QRS changes V3/V4 may be due to LVH but cannot rule out anterior infarct": {
        "MI_OLD": "hedged"
    },
    "*** CONSIDER ACUTE ST ELEVATION MI ***": {"MI_OLD": "excluded"},
    "*** INFERIOR INFARCT - POSSIBLY ACUTE ***": {"MI_OLD": "excluded"},
    "Abnormal ECG": {},
}


class TestTheRules:
    @pytest.mark.parametrize("statement", sorted(ORACLE))
    def test_each_statement_gets_the_status_a_reader_would_give_it(self, statement: str) -> None:
        assert classify(statement) == ORACLE[statement]

    def test_normalising_ignores_case_spacing_and_a_final_full_stop(self) -> None:
        assert normalise("  Sinus  rhythm. ") == normalise("SINUS RHYTHM") == "sinus rhythm"

    def test_every_rule_names_a_known_class_status_and_ambiguity(self) -> None:
        keys = {a.key for a in AMBIGUITIES}
        for rule in RULES:
            assert rule.klass in CLASSES and rule.status in STATUSES
            assert rule.ambiguity is None or rule.ambiguity in keys, rule

    def test_every_ambiguity_is_decided_by_at_least_one_rule(self) -> None:
        assert {a.key for a in AMBIGUITIES} == {r.ambiguity for r in RULES if r.ambiguity}

    def test_a_record_is_positive_when_any_statement_is(self) -> None:
        frame = pd.DataFrame(
            {
                "report_0": ["Sinus rhythm", "Probable atrial fibrillation", "Sinus tachycardia"],
                "report_1": [
                    "with borderline 1st degree A-V block",
                    "Inferior infarct - age undetermined",
                    None,
                ],
            },
            index=["a", "b", "c"],
        )
        for i in range(2, 18):
            frame[f"report_{i}"] = None
        labels = record_labels(record_statuses(frame))
        assert list(labels.loc["a", ["NSR", "IAVB"]]) == [True, True]
        assert list(labels.loc["b"]) == [False, False, False, False, False, True]
        assert not labels.loc["c"].any()
        definite_only = {k: frozenset({"definite"}) for k in CLASSES}
        strict = record_labels(record_statuses(frame), definite_only)
        assert not strict.loc["a", "IAVB"] and not strict.loc["b", "MI_OLD"]

    def test_positive_statuses_are_statuses(self) -> None:
        for klass, statuses in POSITIVE_STATUSES.items():
            assert klass in CLASSES and statuses <= set(STATUSES)


class TestPatientsAndYears:
    def test_one_record_per_patient_whatever_the_row_order(self) -> None:
        patients = pd.Series(["p1", "p1", "p2", "p3", "p3", "p3"], index=list("abcdef"))
        kept = one_per_patient(patients, seed=27)
        assert sorted(patients[kept]) == ["p1", "p2", "p3"]
        shuffled = patients.iloc[np.array([5, 2, 0, 4, 1, 3])]
        again = one_per_patient(shuffled, seed=27)
        assert list(kept) == list(again)

    def test_real_year_bounds_carry_the_anchor_group_and_the_shift(self) -> None:
        # Anchor 2150 lies in real 2011-2013; a tracing three shifted years later
        # lies in real 2014-2016.
        low, high = real_year_bounds(
            pd.Series([2153, 2150]), pd.Series([2150, 2150]), pd.Series(["2011 - 2013"] * 2)
        )
        assert low.tolist() == [2014, 2011] and high.tolist() == [2016, 2013]

    def test_real_eras_keep_only_tracings_surely_inside_a_window(self) -> None:
        frame = pd.DataFrame(
            {
                "patient": ["1", "2", "3", "4", "5"],
                "ecg_time": ["2150-01-01", "2160-06-01", "2150-01-01", "2150-01-01", "2199-01-01"],
            },
            index=list("abcde"),
        )
        patients = pd.DataFrame(
            {
                "subject_id": [1, 2, 3, 5],
                "anchor_year": [2150, 2155, 2150, 2150],
                "anchor_year_group": ["2008 - 2010", "2011 - 2013", "2011 - 2013", "2008 - 2010"],
            }
        )
        era = real_eras(frame, patients)
        # a: 2008-2010 early; b: 2016-2018 late; c: 2011-2013 straddles 2011;
        # d: not in the table; e: 2057-2059, a wrong clock.
        assert era.tolist() == ["early", "late", "unknown", "unknown", "unknown"]

"""CODE-15%, the CODE test set and ZZU pECG: the sample, the readers, the labels, the CLI.

Every expected value here is worked out by hand from the rule the code claims
to follow, on synthetic releases built in ``tmp_path``; none needs a corpus.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import score_rotation
import torch
import train_source

from ecs.code_ecg import (
    code15_sample,
    code_labels,
    proportional_allocation,
    read_code15,
    read_code_test,
)
from ecs.models import ResNet1d
from ecs.rotation import (
    PUBLISHED_ROTATION_DIR,
    _split,
    class_keys,
    rotation_dir_for,
    usable_classes,
)
from ecs.zzu import zzu_frame, zzu_labels


class TestTheAllocation:
    def test_largest_remainders_take_the_spare_draws(self) -> None:
        """A draw that rounds shares badly changes the sample's prevalences from the corpus's."""
        # 15 of (70, 20, 10): exact 10.5, 3.0, 1.5; floors 10, 3, 1 leave one spare,
        # which goes to the first of the two .5 remainders.
        quota = proportional_allocation(pd.Series([70, 20, 10], index=["a", "b", "c"]), 15)
        assert quota.to_dict() == {"a": 11, "b": 3, "c": 1}
        # 10 of (10, 75, 15): exact 1.0, 7.5, 1.5; floors 1, 7, 1 leave one spare, and
        # the largest remainder is not the first stratum's.
        quota = proportional_allocation(pd.Series([10, 75, 15], index=["a", "b", "c"]), 10)
        assert quota.to_dict() == {"a": 1, "b": 8, "c": 1}

    def test_an_exact_share_needs_no_rounding(self) -> None:
        """Proportional allocation with whole shares must reproduce them exactly."""
        quota = proportional_allocation(pd.Series([60, 30, 10], index=["a", "b", "c"]), 10)
        assert quota.to_dict() == {"a": 6, "b": 3, "c": 1}

    def test_a_draw_larger_than_the_corpus_is_refused(self) -> None:
        """Asking for more patients than exist would silently shrink the sample."""
        with pytest.raises(ValueError, match="cannot draw 11 from 10"):
            proportional_allocation(pd.Series([6, 4]), 11)


def _exams() -> tuple[pd.DataFrame, pd.Series]:
    """40 patients: 30 without a flag, 10 with AF; patients 0-9 also hold a 7 s exam."""
    rows = []
    spans = {}
    exam = 100
    for patient in range(40):
        af = patient >= 30
        for seconds in (7, 10) if patient < 10 else (10,):
            rows.append(
                {
                    "exam_id": exam,
                    "patient_id": patient,
                    "1dAVb": False,
                    "RBBB": False,
                    "LBBB": False,
                    "AF": af,
                    "age": 50,
                    "is_male": True,
                }
            )
            spans[exam] = 4000 if seconds == 10 else 2800
            exam += 1
    return pd.DataFrame(rows), pd.Series(spans)


class TestTheCode15Sample:
    def test_the_sample_keeps_the_corpus_prevalence(self) -> None:
        """A sample whose AF share drifts from the corpus's would move every CODE-15% figure."""
        exams, spans = _exams()
        sample = code15_sample(exams, spans, n=8, seed=0)
        # 10 of 40 patients carry AF: 8 drawn in proportion gives 2 with AF, 6 without.
        assert len(sample) == 8
        assert int(sample["AF"].sum()) == 2

    def test_only_ten_second_exams_and_one_per_patient(self) -> None:
        """A seven-second exam cannot fill the window; two exams of a patient would leak."""
        exams, spans = _exams()
        sample = code15_sample(exams, spans, n=40, seed=3)
        assert (spans.loc[sample.index] == 4000).all()
        assert sample["patient_id"].is_unique
        assert len(sample) == 40

    def test_the_seed_fixes_the_draw(self) -> None:
        """The committed scores name exam ids; a draw that moved would orphan them."""
        exams, spans = _exams()
        first = code15_sample(exams, spans, n=8, seed=0).index.tolist()
        assert code15_sample(exams, spans, n=8, seed=0).index.tolist() == first
        assert code15_sample(exams, spans, n=8, seed=1).index.tolist() != first


class TestTheCodeLabels:
    def test_both_spellings_of_a_flag_are_read_and_sinus_rhythm_is_refused(self) -> None:
        """exams.csv writes True/False and gold_standard.csv 1/0; NSR has no CODE flag."""
        flags = pd.DataFrame(
            {
                "1dAVb": ["True", 1, 0],
                "RBBB": ["False", 0, 1],
                "LBBB": [False, 0, 0],
                "AF": [True, 1, 0],
            }
        )
        labels = code_labels(flags)
        assert list(labels.columns) == class_keys()
        assert labels.astype(int).values.tolist() == [
            [0, 1, 0, 0, 1],
            [0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0],
        ]

    def test_sinus_rhythm_leaves_the_code_and_zzu_rows(self) -> None:
        """A refused class read as all-negative would report a coverage of nothing."""
        for corpus in ("code15", "code_test", "zzu"):
            assert usable_classes(corpus) == ["AF", "LBBB", "RBBB", "IAVB"]
        assert usable_classes("mimic") == class_keys()


class TestTheCodeReaders:
    def test_a_short_test_exam_is_cut_to_its_signal_and_filled_at_the_end(
        self, tmp_path: Path
    ) -> None:
        """Seven seconds read as ten with the release's leading zeros would shift every beat."""
        tracing = np.zeros((4096, 12))
        tracing[648 : 648 + 2800] = np.arange(1, 2801)[:, None] * np.arange(1, 13)
        path = tmp_path / "ecg_tracings.hdf5"
        with h5py.File(path, "w") as f:
            f["tracings"] = tracing[None]
        record = read_code_test(f"{path}#0")
        assert record.signal.shape == (12, 4000)
        assert record.signal[0, 0] == 1.0  # the first sample of signal, not padding
        assert record.signal[11, 2799] == 2800 * 12
        assert (record.signal[:, 2800:] == 0).all()
        assert record.lead_names[3:6] == ("aVL", "aVF", "aVR")
        assert record.sampling_rate_hz == 400

    def test_a_code15_exam_keeps_the_release_lead_order(self, tmp_path: Path) -> None:
        """CODE-15% orders aVR, aVL, aVF unlike the test set; swapping them flips the axis."""
        path = tmp_path / "sample.h5"
        tracing = np.tile(np.arange(12, dtype=np.float32), (4000, 1))
        with h5py.File(path, "w") as f:
            f["exam_id"] = np.array([7])
            f["tracings"] = tracing[None]
        record = read_code15(f"{path}#0")
        assert record.signal[:, 0].tolist() == list(range(12))
        assert record.lead_names[3:6] == ("aVR", "aVL", "aVF")


class TestTheZzuLabels:
    def test_codes_are_read_on_their_base_and_statements_of_another_kind_are_not(
        self,
    ) -> None:
        """Incomplete RBBB, 'Suggests' codes or a lost modifier must not change a class."""
        column = pd.Series(
            [
                "'I106';'C21'",  # RBBB
                "'I105'",  # incomplete RBBB: not the class
                "'E50+Modifier346'",  # AF with a modifier
                "'Suggests104'",  # a 'suggests' statement numbered like LBBB
                "'First-degree AV block';'L147'",  # IAVB with no AHA code
                "'H82'",  # prolonged PR, AHA 82
                "'I104'",  # LBBB
            ]
        )
        labels = zzu_labels(column)
        assert labels.astype(int).values.tolist() == [
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0],
        ]

    def test_the_frame_keeps_one_twelve_lead_ten_second_record_per_child(
        self, tmp_path: Path
    ) -> None:
        """Nine-lead or five-second records cannot be read; two of a child's would leak."""
        pd.DataFrame(
            {
                "Filename": ["P00/P1/P1_E01", "P00/P1/P1_E02", "P00/P2/P2_E01", "P00/P3/P3_E01"],
                "ECG_ID": ["P1_E01", "P1_E02", "P2_E01", "P3_E01"],
                "Patient_ID": ["P1", "P1", "P2", "P3"],
                "Age": ["365d", "730d", "3652d", "100d"],
                "Gender": ["'Male'", "'Male'", "'Female'", "'Male'"],
                "Sampling_point": [10000, 10000, 15000, 4000],
                "Lead": [12, 12, 9, 12],
                "AHA_code": ["'I106'", "'A1'", "'E50'", "'A1'"],
            }
        ).to_csv(tmp_path / "AttributesDictionary.csv", index=False)
        frame, labels = zzu_frame(tmp_path, tmp_path / "records")
        assert list(frame["patient"]) == ["P1"]
        assert frame["age"].iloc[0] in (pytest.approx(1.0, abs=0.01), pytest.approx(2.0, abs=0.01))
        assert frame["path"].iloc[0].startswith(str(tmp_path / "records" / "Child_ecg" / "P00"))


class TestTheTargetOnlySplit:
    def test_a_target_only_corpus_is_all_test_and_capped_by_patient(self) -> None:
        """ZZU in a training part would make the children a source of the adult rotation."""
        frame = pd.DataFrame({"patient": [str(i // 2) for i in range(20_000)]})
        frame.index = pd.Index([str(i) for i in range(20_000)])
        parts = _split(frame, target_only=True)["part"]
        assert set(parts) == {"test", "unused"}
        assert int((parts == "test").sum()) == 8000
        kept = frame.loc[parts == "test", "patient"]
        assert kept.value_counts().eq(2).all()  # whole patients only


class TestTheRotationDirectoryGuard:
    def test_a_corpus_outside_the_five_is_refused_the_published_directory(self) -> None:
        """A CODE-15% model written under results/rotation would overwrite the published run."""
        with pytest.raises(ValueError, match="code15 is not in the published rotation"):
            rotation_dir_for(["code15"], None)
        assert rotation_dir_for(["ptbxl", "georgia"], None) == PUBLISHED_ROTATION_DIR
        assert rotation_dir_for(["code15"], Path("elsewhere")) == Path("elsewhere")

    def test_train_source_refuses_without_rotation_dir_and_reads_device(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Review of PR 11: --source mimic without --rotation-dir wrote beside the published run."""
        with pytest.raises(SystemExit):
            train_source.arguments(["--source", "mimic"])
        assert "--rotation-dir" in capsys.readouterr().err
        source, settings, where, _ = train_source.arguments(
            ["--source", "code15", "--rotation-dir", "out", "--device", "mps"]
        )
        assert (source, settings.device, where) == ("code15", "mps", Path("out"))
        _, published, where, _ = train_source.arguments(["--source", "ptbxl"])
        assert (published.device, where) == ("cpu", PUBLISHED_ROTATION_DIR)

    def test_score_rotation_refuses_a_new_corpus_without_rotation_dir(self) -> None:
        """Scoring ZZU with the published models' directory would add files to the published run."""
        with pytest.raises(SystemExit):
            score_rotation.arguments(["--corpora", "zzu"])
        sources, corpora, device, where, _ = score_rotation.arguments(
            [
                "--sources",
                "ptbxl,code15",
                "--corpora",
                "zzu",
                "--rotation-dir",
                "r",
                "--device",
                "mps",
            ]
        )
        assert (sources, corpora, device, where) == (["ptbxl", "code15"], ["zzu"], "mps", Path("r"))

    def test_the_scorer_reads_its_model_from_the_rotation_dir(self, tmp_path: Path) -> None:
        """A scorer reading the published directory would score with the wrong model."""
        torch.manual_seed(0)
        model = ResNet1d(n_classes=len(class_keys()))
        directory = tmp_path / "code15"
        directory.mkdir()
        torch.save(model.state_dict(), directory / "model.pt")
        config = {"standardisation": {"centre_mv": 0.5, "scale_mv": 2.0}, "usable_classes": []}
        (directory / "config.json").write_text(json.dumps(config))
        scorer = score_rotation.Scorer("code15", tmp_path, device="cpu")
        x = np.random.default_rng(0).normal(size=(3, 12, 5000)).astype(np.float32)
        model.eval()
        with torch.no_grad():
            expected = torch.sigmoid(model(torch.from_numpy((x - 0.5) / 2.0))).numpy()
        assert np.allclose(scorer.probabilities(x), expected, atol=1e-6)

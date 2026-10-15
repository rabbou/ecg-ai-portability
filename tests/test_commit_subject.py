"""The commit-msg hook refuses the two subjects reviews kept flagging, and nothing else."""

from __future__ import annotations

from pathlib import Path

import pytest
from check_commit_subject import main, problems, subject_of


@pytest.mark.parametrize(
    "subject",
    [
        "Rebuild the random arm under its seed and requote section S2.10 (#14)",
        "Merge pull request #79 from rabbou/tache/pnds-amylose",
        "Fix the label_free denominator: weigh false flags by 1 - prevalence",
    ],
)
def test_a_flat_imperative_subject_passes(subject: str) -> None:
    assert problems(subject) == []


def test_a_subject_over_72_characters_is_refused() -> None:
    # eb53ca9 on master, 81 characters.
    subject = "Index results/ and reports/transfer/, and state the tree audit's exceptions (#16)"
    assert len(subject) == 81
    assert problems(subject) and "81 characters" in problems(subject)[0]
    assert problems("x" * 72) == []


@pytest.mark.parametrize("subject", ["Study: rebuild the results", "fix(ci): run full history"])
def test_a_framing_colon_is_refused(subject: str) -> None:
    assert problems(subject) == ["the subject opens on a label and a colon; start with the verb"]


def test_the_subject_is_the_first_line_that_is_not_a_comment(tmp_path: Path) -> None:
    message = tmp_path / "COMMIT_EDITMSG"
    message.write_text("# Please enter\n\nStudy: rebuild\n\nbody: with a colon\n")
    assert subject_of(message.read_text()) == "Study: rebuild"
    assert main(["hook", str(message)]) == 1
    message.write_text("Rebuild the results\n\nbody: with a colon\n")
    assert main(["hook", str(message)]) == 0

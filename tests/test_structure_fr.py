"""STRUCTURE.fr.md carries every number of STRUCTURE.md, and no other, under the same sections.

The French text and its tables are translated by hand, not rendered, so a re-render of
STRUCTURE.md that moves a figure fails here until the translation follows.  The number
rules are those of REPORT.fr.md.
"""

from __future__ import annotations

import re
from collections import Counter

from test_report_fr import english_numbers, french_numbers

from ecs.config import REPO_ROOT

DOC = REPO_ROOT / "STRUCTURE.md"
DOC_FR = REPO_ROOT / "STRUCTURE.fr.md"


def test_the_french_text_carries_the_same_numbers() -> None:
    # The table markers of STRUCTURE.md (``table:a1``) are not text.
    english = english_numbers(re.sub(r"<!--.*?-->", "", DOC.read_text()))
    french = french_numbers(DOC_FR.read_text())
    assert french - english == Counter() and english - french == Counter(), (
        f"only in STRUCTURE.fr.md: {dict(french - english)}; "
        f"only in STRUCTURE.md: {dict(english - french)}"
    )


def test_the_french_text_keeps_the_sections() -> None:
    def sections(text: str) -> list[tuple[str, str]]:
        return re.findall(r"^(#{2,3}) (\d[\d.]*)?", text, flags=re.M)

    assert sections(DOC_FR.read_text()) == sections(DOC.read_text())


def test_the_french_text_shows_the_french_figures() -> None:
    """Each figure of STRUCTURE.md appears in the French text as its ``_fr`` twin, on disk."""
    english = re.findall(r"!\[[^]]*\]\(([^)]+)\)", DOC.read_text())
    french = re.findall(r"!\[[^]]*\]\(([^)]+)\)", DOC_FR.read_text())
    assert english, "STRUCTURE.md shows no figure"
    assert french == [path.replace(".png", "_fr.png") for path in english]
    assert all((REPO_ROOT / path).exists() for path in english + french)

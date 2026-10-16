"""REPORT.fr.md carries every number and figure of REPORT.md, and no other.

The French text is translated by hand, not rendered, so a re-render of
REPORT.md that moves a figure fails here until the translation follows.
French prose writes 5,3 and 1 903 (a narrow or a plain no-break space between groups)
where English writes 5.3 and 1,903; the reference list stays in English in
both, so it is read with the English rule.
"""

from __future__ import annotations

import re
from collections import Counter

from paper_render import REPORT

REPORT_FR = REPORT.with_name("REPORT.fr.md")

# Digits, then thousands groups, then any run of decimal or dotted parts:
# 1,903 · 5.3 · v1.1.1 · [2,3] · 1 903 · 5,3.
EN_NUMBER = re.compile(r"\d+(?:,\d{3}(?!\d))*(?:[.,]\d+)*")
FR_NUMBER = re.compile(r"\d+(?:[\u00a0\u202f]\d{3}(?!\d))*(?:[.,]\d+)*")
REFERENCES_FR = re.compile(r"^## Références[ \t]*$", flags=re.M)
IMAGE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def english_numbers(text: str) -> Counter[str]:
    found = (re.sub(r",(?=\d{3}(?!\d))", "", n) for n in EN_NUMBER.findall(text))
    return Counter(n.replace(",", ".") for n in found)


def french_numbers(text: str) -> Counter[str]:
    parts = REFERENCES_FR.split(text)
    assert len(parts) == 2, f"expected one '## Références' heading, found {len(parts) - 1}"
    body, references = parts
    found = (re.sub(r"[\u00a0\u202f]", "", n).replace(",", ".") for n in FR_NUMBER.findall(body))
    return Counter(found) + english_numbers(references)


def test_the_french_report_carries_the_same_numbers() -> None:
    english = english_numbers(REPORT.read_text())
    french = french_numbers(REPORT_FR.read_text())
    assert french - english == Counter() and english - french == Counter(), (
        f"only in REPORT.fr.md: {dict(french - english)}; "
        f"only in REPORT.md: {dict(english - french)}"
    )


def test_the_french_report_shows_the_french_figures_in_the_same_order() -> None:
    """Each English figure has a French twin, ``<name>_fr.png``, and that is the one shown."""
    english = IMAGE.findall(REPORT.read_text())
    french = IMAGE.findall(REPORT_FR.read_text())
    assert french == [path.replace(".png", "_fr.png") for path in english]
    for path in french:
        assert (REPORT.parent / path).is_file(), path


def test_the_french_report_keeps_the_sections() -> None:
    def sections(text: str) -> list[tuple[str, str]]:
        return re.findall(r"^(#{2,3}) (\d[\d.]*)?", text, flags=re.M)

    assert sections(REPORT_FR.read_text()) == sections(REPORT.read_text())


def test_the_rules_tell_french_and_english_numbers_apart() -> None:
    assert english_numbers("1,903 at 5.3 [2,3] v1.1.1") == Counter(
        {"1903": 1, "5.3": 1, "2.3": 1, "1.1.1": 1}
    )
    assert french_numbers("1 903 à 5,3 [2,3] v1.1.1\n## Références\n19,955") == Counter(
        {"1903": 1, "5.3": 1, "2.3": 1, "1.1.1": 1, "19955": 1}
    )


def test_either_no_break_space_separates_french_thousands() -> None:
    text = "1\u202f903 et 1\u00a0903, 5,3\n## Références\n1. Doe, 2021;3:4."
    assert french_numbers(text) == Counter({"1903": 2, "5.3": 1, "1": 1, "2021": 1, "3": 1, "4": 1})


def test_the_references_are_cut_at_their_heading_only() -> None:
    text = "voir « ## Références » plus bas, 1\u202f903\n## Références\n1. Doe 2021."
    assert french_numbers(text) == Counter({"1903": 1, "1": 1, "2021": 1})

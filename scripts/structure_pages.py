"""Render the tables and the figure of STRUCTURE.md and its supplement from the result files.

Each table is written between its ``<!-- table:<name> -->`` and ``<!-- /table -->``
lines, in STRUCTURE.md or in STRUCTURE_SUPPLEMENT.md; every table the renderer
knows sits in exactly one block of the two.  The figure,
``structure_sensitivity.png`` and its French twin, plots the composite
sensitivity per model at the inpatient threshold and at each weighting.

Reads only committed files: ``results/structure.json``, its CSV and calibration
file, and ``results/structure_power.json``.

Usage: .venv/bin/python scripts/structure_pages.py
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from figure_text import LANGUAGES, SUFFIX, percent  # noqa: E402

from ecs.config import REPO_ROOT, RESULTS_DIR  # noqa: E402
from ecs.echonext import COMPOSITE  # noqa: E402
from ecs.structure import COMPOSITE_BAND  # noqa: E402
from ecs.structure_report import composite_points, load, tables  # noqa: E402

DOCS = (REPO_ROOT / "STRUCTURE.md", REPO_ROOT / "STRUCTURE_SUPPLEMENT.md")
BLOCK = re.compile(r"(<!-- table:(\w+) -->\n)(.*?)(<!-- /table -->)", re.DOTALL)
FIGURES = RESULTS_DIR / "figures"
MUTED, GRID = "#52514e", "#eeeeee"
# One colour and one marker per threshold, so no threshold rests on colour alone.
THRESHOLD_STYLE = {
    "plain": ("#a3a29d", "v"),
    "age_sex": ("#eb6834", "s"),
    "tabular": ("#1baf7a", "D"),
    "embedding": ("#2a78d6", "o"),
}

WORDS: dict[str, dict[str, str]] = {
    "en": {
        "title": "Composite sensitivity for SHD, by threshold",
        "x": "Sensitivity, with 95% Wilson interval",
        "band": "85% to 95%",
        "panel.outpatient": "Outpatients, {cases} with SHD",
        "panel.emergency": "Emergency patients, {cases} with SHD",
        "threshold.plain": "inpatient threshold",
        "threshold.age_sex": "weighted on age, sex",
        "threshold.tabular": "weighted on ECG measurements",
        "threshold.embedding": "weighted on embedding",
        "arm.resnet": "ResNet trained on EchoNext",
        "arm.random_init": "ResNet at random init., probes",
        "arm.ecgfounder": "ECGFounder frozen, probes",
        "arm.echonext_mini": "EchoNext mini-model",
        "arm.ecgfounder_ft_s0": "ECGFounder fine-tuned, seed 0",
        "arm.ecgfounder_ft_s1": "ECGFounder fine-tuned, seed 1",
        "arm.ecgfounder_ft_s2": "ECGFounder fine-tuned, seed 2",
    },
    "fr": {
        "title": "Sensibilité composite pour la cardiopathie structurelle, par seuil",
        "x": "Sensibilité, avec intervalle de Wilson à 95 %",
        "band": "85 % à 95 %",
        "panel.outpatient": "Patients ambulatoires, {cases} avec cardiopathie",
        "panel.emergency": "Patients des urgences, {cases} avec cardiopathie",
        "threshold.plain": "seuil des patients hospitalisés",
        "threshold.age_sex": "pondéré sur l'âge et le sexe",
        "threshold.tabular": "pondéré sur les mesures de l'ECG",
        "threshold.embedding": "pondéré sur la représentation",
        "arm.resnet": "ResNet entraîné sur EchoNext",
        "arm.random_init": "ResNet à l'init. aléatoire, sondes",
        "arm.ecgfounder": "ECGFounder figé, sondes",
        "arm.echonext_mini": "Mini-modèle EchoNext",
        "arm.ecgfounder_ft_s0": "ECGFounder affiné, graine 0",
        "arm.ecgfounder_ft_s1": "ECGFounder affiné, graine 1",
        "arm.ecgfounder_ft_s2": "ECGFounder affiné, graine 2",
    },
}


def write_pages(rendered: dict[str, str]) -> None:
    """Replace each table block of the two documents with its rendering."""
    texts = {doc: doc.read_text() for doc in DOCS}
    names = [m.group(2) for text in texts.values() for m in BLOCK.finditer(text)]
    if sorted(names) != sorted(rendered):
        raise ValueError(f"the documents hold {sorted(names)}, the renderer {sorted(rendered)}")
    for doc, text in texts.items():
        doc.write_text(
            BLOCK.sub(lambda m: f"{m.group(1)}{rendered[m.group(2)]}\n{m.group(4)}", text)
        )


def figure_sensitivity(
    summary: dict[str, Any], coverage: pd.DataFrame, out: Path, lang: str
) -> Path:
    text = WORDS[lang]
    contexts = summary["verdict_contexts"]
    arms = summary["arms"]
    figure, axes = plt.subplots(1, len(contexts), figsize=(11, 5.4), sharey=True)
    offsets = dict(zip(THRESHOLD_STYLE, (-0.27, -0.09, 0.09, 0.27), strict=True))
    for axis, context in zip(axes, contexts, strict=True):
        points = composite_points(summary, coverage, context)
        low, high = COMPOSITE_BAND
        axis.axvspan(low, high, color=GRID, zorder=0)
        axis.axvline(0.9, color=MUTED, linewidth=1, linestyle=":", zorder=1)
        for row, arm in enumerate(arms):
            for threshold, (colour, marker) in THRESHOLD_STYLE.items():
                value, lower, upper, _ = points[arm][threshold]
                y = row + offsets[threshold]
                axis.plot([lower, upper], [y, y], color=colour, linewidth=1.4, zorder=2)
                axis.scatter(
                    value, y, color=colour, marker=marker, s=26, edgecolors="white", zorder=3
                )
        cases = int(
            coverage.loc[
                (coverage["context"] == context)
                & (coverage["method"] == "plain")
                & (coverage["label"] == COMPOSITE)
                & (coverage["arm"] == arms[0]),
                "n_pos",
            ].iloc[0]
        )
        axis.set_title(text[f"panel.{context}"].format(cases=cases), fontsize=10)
        axis.set_xlim(0.6, 1.0)
        axis.set_xlabel(text["x"])
        axis.xaxis.set_major_formatter(lambda v, _: percent(v, 0, lang))
        axis.grid(axis="x", color=GRID, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(
            (low + high) / 2,
            -0.52,
            text["band"],
            color=MUTED,
            fontsize=8,
            ha="center",
            va="bottom",
        )
    axes[0].set_yticks(range(len(arms)), [text[f"arm.{a}"] for a in arms], fontsize=9)
    axes[0].set_ylim(len(arms) - 0.3, -0.85)
    handles = [
        plt.Line2D([], [], color=c, marker=m, label=text[f"threshold.{k}"])
        for k, (c, m) in THRESHOLD_STYLE.items()
    ]
    figure.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8)
    figure.suptitle(text["title"], fontsize=11)
    figure.tight_layout(rect=(0, 0.06, 1, 0.96))
    figure.savefig(out, dpi=150)
    plt.close(figure)
    return out


def main() -> None:
    summary, coverage, calibration = load()
    write_pages(tables(summary, coverage, calibration))
    out = FIGURES
    out.mkdir(parents=True, exist_ok=True)
    for lang in LANGUAGES:
        suffix = SUFFIX[lang]
        path = figure_sensitivity(
            summary, coverage, out / f"structure_sensitivity{suffix}.png", lang
        )
        print(path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()

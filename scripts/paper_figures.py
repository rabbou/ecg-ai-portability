"""Draw the three figures of REPORT.md from ``results/echonext_clinical.json``.

``fig_curve.png``: for the trained network, the share of patients with SHD
caught against the share of patients without SHD flagged, inpatients and
outpatients, with the point the inpatient threshold lands on in each setting.

``fig_patients.png``: what the inpatient threshold does to 100 outpatients with
SHD and 100 without, for each model, alone and with the second threshold.

``fig_repair.png``: the threshold refitted on labelled outpatients, sensitivity
and specificity against the number labelled, with the spread over draws.

``--lang fr`` draws the same three with French text, as ``<name>_fr.png``, for
REPORT.fr.md.

Usage: uv run python scripts/paper_figures.py [--lang en|fr]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from figure_text import LANGUAGES, SUFFIX  # noqa: E402

from ecs.config import RESULTS_DIR  # noqa: E402

FIGURES = RESULTS_DIR / "figures"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#eeeeee"
# Fixed slots: a setting or an outcome keeps its colour in every figure, and
# each also carries a dash, a marker or a printed label.
INPATIENT, OUTPATIENT = "#2a78d6", "#eb6834"
RIGHT, DEFERRED, WRONG = "#2a78d6", "#c9c7c1", "#eb6834"
ARMS = ("resnet", "echonext_mini", "ecgfounder", "random_init")
WORDS: dict[str, dict[str, str]] = {
    "en": {
        "arm.resnet": "Network trained here",
        "arm.echonext_mini": "EchoNext mini-model",
        "arm.ecgfounder": "ECGFounder",
        "arm.random_init": "Untrained floor",
        "inpatient": "Inpatients",
        "outpatient": "Outpatients",
        "point": "{y} of 100 with SHD caught\n{x} of 100 without SHD flagged",
        "curve.x": "Patients without SHD flagged, per 100",
        "curve.y": "Patients with SHD caught, per 100",
        "plain": "one threshold",
        "perlabel": "two thresholds",
        "group.ill": "Per 100 outpatients with SHD",
        "group.healthy": "Per 100 outpatients without SHD",
        "ill.right": "flagged, no reader",
        "ill.deferred": "sent to a reader",
        "ill.wrong": "missed",
        "healthy.right": "cleared, no reader",
        "healthy.deferred": "sent to a reader",
        "healthy.wrong": "flagged, no reader",
        "repair.sensitivity": "With SHD caught",
        "repair.specificity": "Without SHD flagged",
        "repair.in_sample": "\n({n} SHD)",
        "repair.none": "\n(none)",
        "repair.x": "Labelled outpatients the threshold is refitted on",
        "repair.y": "Per 100 outpatients of each group",
    },
    "fr": {
        "arm.resnet": "Réseau entraîné ici",
        "arm.echonext_mini": "Mini-modèle EchoNext",
        "arm.ecgfounder": "ECGFounder",
        "arm.random_init": "Plancher non entraîné",
        "inpatient": "Patients hospitalisés",
        "outpatient": "Patients ambulatoires",
        "point": "{y} avec CS détectés sur 100\n{x} sans CS signalés sur 100",
        "curve.x": "Patients sans CS signalés, pour 100",
        "curve.y": "Patients avec CS détectés, pour 100",
        "plain": "un seuil",
        "perlabel": "deux seuils",
        "group.ill": "Pour 100 patients ambulatoires avec CS",
        "group.healthy": "Pour 100 patients ambulatoires sans CS",
        "ill.right": "signalés, sans lecteur",
        "ill.deferred": "envoyés à un lecteur",
        "ill.wrong": "manqués",
        "healthy.right": "écartés, sans lecteur",
        "healthy.deferred": "envoyés à un lecteur",
        "healthy.wrong": "signalés, sans lecteur",
        "repair.sensitivity": "Avec CS détectés",
        "repair.specificity": "Sans CS signalés",
        "repair.in_sample": "\n({n} CS)",
        "repair.none": "\n(aucun)",
        "repair.x": "Patients ambulatoires étiquetés sur lesquels le seuil est réajusté",
        "repair.y": "Pour 100 patients ambulatoires de chaque groupe",
    },
}


def read() -> dict[str, Any]:
    result: dict[str, Any] = json.loads((RESULTS_DIR / "echonext_clinical.json").read_text())
    return result


def style(axis: Any) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(MUTED)
    axis.tick_params(colors=MUTED, labelsize=8)
    axis.grid(color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)


def curve(result: dict[str, Any], path: Path, lang: str) -> None:
    text = WORDS[lang]
    roc = result["roc"]["resnet"]
    arm = result["arms"]["resnet"]
    fig, axis = plt.subplots(figsize=(5.2, 4.4), dpi=200)
    style(axis)
    for context, colour, dash in (
        ("inpatient", INPATIENT, "-"),
        ("outpatient", OUTPATIENT, "--"),
    ):
        name = text[context]
        flagged = [100 * (1 - s) for s in roc[context]]
        axis.plot(
            flagged, [100 * s for s in roc["sensitivity"]], dash, color=colour, lw=2, label=name
        )
        x, y = 100 * (1 - arm[context]["specificity"]), 100 * arm[context]["sensitivity"]
        axis.plot(x, y, "o", ms=9, color=colour, markeredgecolor="white", markeredgewidth=2)
        axis.annotate(
            text["point"].format(y=f"{y:.0f}", x=f"{x:.0f}"),
            (x, y),
            xytext=(10, -30) if context == "outpatient" else (-150, 6),
            textcoords="offset points",
            fontsize=8,
            color=INK,
        )
    axis.axhline(90, color=MUTED, lw=1, ls=":")
    axis.text(101, 90, "90", fontsize=7, color=MUTED, va="center")
    axis.set_xlim(0, 100)
    axis.set_ylim(0, 102)
    axis.set_xlabel(text["curve.x"], fontsize=9, color=INK)
    axis.set_ylabel(text["curve.y"], fontsize=9, color=INK)
    axis.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def largest_remainder(shares: list[float]) -> list[int]:
    """Shares that sum to one, as whole counts per 100 that sum to 100: each is rounded
    down, and the points left go to the largest remainders."""
    exact = [100 * s for s in shares]
    counts = [int(x) for x in exact]
    order = sorted(range(len(exact)), key=lambda i: exact[i] - counts[i], reverse=True)
    for i in order[: 100 - sum(counts)]:
        counts[i] += 1
    return counts


def patients(result: dict[str, Any], path: Path, lang: str) -> None:
    text = WORDS[lang]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), dpi=200, sharey=True)
    rows = [(arm, method) for arm in ARMS for method in ("plain", "perlabel")]
    labels = [f"{text['arm.' + arm]}, {text[method]}" for arm, method in rows]
    for axis, group in ((axes[0], "ill"), (axes[1], "healthy")):
        names = [text[f"{group}.{outcome}"] for outcome in ("right", "deferred", "wrong")]
        style(axis)
        axis.grid(False)
        for i, (arm, method) in enumerate(rows):
            shares = result["arms"][arm]["outpatient"]["outcomes"][method][group]
            keys = ("right_alone", "deferred", "wrong_alone")
            counts = largest_remainder([shares[k]["share"] for k in keys])
            left = 0.0
            for key, colour, shown in zip(keys, (RIGHT, DEFERRED, WRONG), counts, strict=True):
                width = 100 * shares[key]["share"]
                axis.barh(
                    i, width, left=left, color=colour, edgecolor="white", linewidth=2, height=0.75
                )
                if width >= 6:
                    ink = "white" if colour != DEFERRED else INK
                    axis.text(
                        left + width / 2,
                        i,
                        f"{shown}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color=ink,
                    )
                left += width
        axis.set_xlim(0, 100)
        axis.set_title(text[f"group.{group}"], fontsize=9, color=INK, loc="left")
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (RIGHT, DEFERRED, WRONG)]
        axis.legend(
            handles,
            names,
            frameon=False,
            fontsize=7,
            ncol=2,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.06),
        )
        axis.set_xticks([])
        axis.spines[["bottom"]].set_visible(False)
    axes[0].set_yticks(range(len(rows)), labels, fontsize=7)
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def repair(result: dict[str, Any], path: Path, lang: str) -> None:
    text = WORDS[lang]
    ladder = result["ladder"]["resnet"]
    fig, axis = plt.subplots(figsize=(5.6, 4.0), dpi=200)
    style(axis)
    x = [row["labels"] for row in ladder]
    for key, colour, dash in (
        ("sensitivity", OUTPATIENT, "-"),
        ("specificity", INPATIENT, "--"),
    ):
        name = text[f"repair.{key}"]
        # Patients without SHD flagged is one minus the specificity, so its band flips ends.
        flip = key == "specificity"
        mean = [100 * (1 - row[key]["mean"] if flip else row[key]["mean"]) for row in ladder]
        low = [100 * (1 - row[key]["p90"] if flip else row[key]["p10"]) for row in ladder]
        high = [100 * (1 - row[key]["p10"] if flip else row[key]["p90"]) for row in ladder]
        axis.fill_between(x, low, high, color=colour, alpha=0.15, linewidth=0)
        axis.plot(x, mean, dash, color=colour, lw=2, marker="o", ms=5, label=name)
    axis.axhline(90, color=MUTED, lw=1, ls=":")
    ticks = [
        f"{row['labels']}"
        + (
            text["repair.in_sample"].format(n=f"{row['ill_in_sample']['mean']:.0f}")
            if row["labels"]
            else text["repair.none"]
        )
        for row in ladder
    ]
    axis.set_xticks(x, ticks, fontsize=7)
    axis.set_ylim(0, 102)
    axis.set_xlabel(text["repair.x"], fontsize=9, color=INK)
    axis.set_ylabel(text["repair.y"], fontsize=9, color=INK)
    axis.legend(frameon=False, fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", default="en", choices=LANGUAGES)
    lang = parser.parse_args(argv).lang
    result = read()
    out, suffix = FIGURES, SUFFIX[lang]
    for draw, path in (
        (curve, out / f"fig_curve{suffix}.png"),
        (patients, out / f"fig_patients{suffix}.png"),
        (repair, out / f"fig_repair{suffix}.png"),
    ):
        draw(result, path, lang)
        print(path.relative_to(RESULTS_DIR.parent))


if __name__ == "__main__":
    main()

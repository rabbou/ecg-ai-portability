"""The words the three report figures carry, in English and in French.

Each figure is drawn by one function that takes a language, so both sets come
from the same numbers through the same code; only the strings below and the way
a number is printed differ.  Every key present in one language is present in the
other, which a test holds.
"""

from __future__ import annotations

LANGUAGES = ("en", "fr")
# File name suffix per language: the English set keeps the names the report uses.
SUFFIX = {"en": "", "fr": "_fr"}
# French sets a non-breaking space before a per cent sign and between thousands.
_NBSP = " "


def number(value: float, digits: int, lang: str) -> str:
    """A decimal as the language prints it: 0.57 or 0,57."""
    text = f"{value:.{digits}f}"
    return text.replace(".", ",") if lang == "fr" else text


def count(value: float, lang: str) -> str:
    """A whole count with its thousands separator: 1,648 or 1 648."""
    text = f"{value:,.0f}"
    return text.replace(",", _NBSP) if lang == "fr" else text


def percent(share: float, digits: int, lang: str) -> str:
    """A share of one as a percentage: 73% or 73 %."""
    body = number(share * 100, digits, lang)
    return f"{body}{_NBSP}%" if lang == "fr" else f"{body}%"


WORDS: dict[str, dict[str, str]] = {
    "en": {
        # The three corrections, as panel titles: the name, then what it does.
        "correction.none": "pooled, no correction\none threshold for both classes",
        "correction.mondrian": "class-conditional\none threshold per class",
        "correction.weighted": "label-shift weighted\ntarget prior estimated",
        "group.overall": "every tracing",
        "group.0": "no infarction",
        "group.1": "infarction",
        "corpus.ptbxl": "PTB-XL (calibrated here)",
        "corpus.sph": "Shandong",
        "corpus.acs": "Chongqing",
        "coverage.panel": "{name}\n{n} tracings, {prevalence} infarction",
        "coverage.ylabel": "share whose set holds the true label",
        "coverage.xlabel": "confidence asked for",
        "coverage.title": "Coverage per hospital and correction",
        "coverage.quoted": "Share of infarctions inside the 90% set on PTB-XL: {quoted}.",
        "coverage.spread": (
            "Dashed line: the requested level. Bars: mean over {draws} calibration draws "
            "on PTB-XL, whiskers one standard deviation."
        ),
        "coverage.weighted": (
            "The weighted thresholds use each corpus's estimated class mix:\n{estimated}."
        ),
        "coverage.estimate": "{corpus} {estimated} estimated for {true} true",
        # The three schemes of the outcome table: the name, then its target.
        "scheme.plain": "Single tuned threshold\n90% sensitivity, no deferral",
        "scheme.pooled": "Pooled conformal calibration\n90% coverage, all cases",
        "scheme.perlabel": "Class-conditional calibration\n90% coverage within each label",
        "class.0": "non-MI",
        "class.1": "MI",
        "thresholds.legend": "{name} (n={n})",
        "thresholds.deferred": "deferred",
        "thresholds.labelled.0": "labelled non-MI",
        "thresholds.labelled.1": "labelled MI",
        "thresholds.box": "MI missed  {missed}\nfalse alarms  {alarms}\ndeferred  {deferred}",
        "thresholds.ylabel": "% of that class",
        "thresholds.xlabel": "model score for MI",
        "thresholds.note.rates": (
            "MI missed: share of MI cases given the non-MI label alone.   "
            "False alarms: share of non-MI cases given the MI label alone."
        ),
        "thresholds.note.deferred": (
            "Deferred: share of all tracings returning both labels or neither, "
            "sent to a specialist."
        ),
        "thresholds.note.halves": (
            "Histograms show all of fold 10; the rates are means over the test halves."
        ),
        "outcomes.panel.1": "MI cases (mean {n} per draw)",
        "outcomes.panel.0": "non-MI cases (mean {n} per draw)",
        "outcomes.xlabel": "share of cases carrying this label (%)",
        "outcome.correct": "correct label returned",
        "outcome.deferred": "deferred to a specialist",
        "outcome.wrong": "incorrect label returned",
        "outcomes.note": (
            "PTB-XL fold 10, {draws} patient-level calibration draws.\nThe single threshold "
            "and class-conditional calibration meet at an MI miss rate of about {matched} by "
            "construction; pooled calibration is matched to neither and lands at {pooled}."
        ),
    },
    "fr": {
        "correction.none": "commune, sans correction\nun seuil pour les deux classes",
        "correction.mondrian": "par classe\nun seuil pour chaque classe",
        "correction.weighted": "pondérée par la prévalence\nprévalence cible estimée",
        "group.overall": "tous les tracés",
        "group.0": "sans infarctus",
        "group.1": "infarctus",
        "corpus.ptbxl": "PTB-XL (calibré ici)",
        "corpus.sph": "Shandong",
        "corpus.acs": "Chongqing",
        "coverage.panel": "{name}\n{n} tracés, {prevalence} d'infarctus",
        "coverage.ylabel": "part dont l'ensemble contient le bon diagnostic",
        "coverage.xlabel": "confiance demandée",
        "coverage.title": "Couverture par hôpital et par correction",
        "coverage.quoted": (
            "Part des infarctus dont l'ensemble à 90 % contient le diagnostic, "
            "sur PTB-XL : {quoted}."
        ),
        "coverage.spread": (
            "Tirets : le niveau demandé. Barres : moyenne sur {draws} tirages de "
            "calibration sur PTB-XL, moustaches à un écart type."
        ),
        "coverage.weighted": (
            "Les seuils pondérés partent de la prévalence estimée "
            "dans chaque corpus\u00a0:\n{estimated}."
        ),
        "coverage.estimate": "{corpus} {estimated} estimée pour {true} réelle",
        "scheme.plain": "Seuil unique ajusté\nsensibilité de 90 %, aucun renvoi",
        "scheme.pooled": "Calibration conforme commune\ncouverture de 90 %, tous cas confondus",
        "scheme.perlabel": "Calibration conforme par classe\ncouverture de 90 % dans chaque classe",
        "class.0": "sans IDM",
        "class.1": "IDM",
        "thresholds.legend": "{name} (n = {n})",
        "thresholds.deferred": "renvoi",
        "thresholds.labelled.0": "classé sans IDM",
        "thresholds.labelled.1": "classé IDM",
        "thresholds.box": "IDM manqués  {missed}\nfausses alertes  {alarms}\nrenvois  {deferred}",
        "thresholds.ylabel": "% de la classe",
        "thresholds.xlabel": "score du modèle pour l'IDM",
        "thresholds.note.rates": (
            "IDM manqués : part des IDM classés sans IDM, seul diagnostic rendu.   "
            "Fausses alertes : part des cas sans IDM classés IDM."
        ),
        "thresholds.note.deferred": (
            "Renvois : part de tous les tracés qui reçoivent les deux diagnostics ou "
            "aucun, confiés à un spécialiste."
        ),
        "thresholds.note.halves": (
            "Les histogrammes montrent tout le pli 10 ; les taux sont des moyennes sur "
            "les moitiés de test."
        ),
        "outcomes.panel.1": "Cas d'IDM ({n} en moyenne par tirage)",
        "outcomes.panel.0": "Cas sans IDM ({n} en moyenne par tirage)",
        "outcomes.xlabel": "part des cas portant ce diagnostic (%)",
        "outcome.correct": "bon diagnostic rendu",
        "outcome.deferred": "renvoyé à un spécialiste",
        "outcome.wrong": "mauvais diagnostic rendu",
        "outcomes.note": (
            "PTB-XL, pli 10, {draws} tirages de calibration par patient.\nLe seuil unique et "
            "la calibration par classe manquent environ {matched} des IDM, par construction ; "
            "la calibration commune n'est alignée sur aucun des deux et en manque {pooled}."
        ),
    },
}


def words(lang: str) -> dict[str, str]:
    """The catalogue for one language; an unknown language is an error, not English."""
    if lang not in WORDS:
        raise ValueError(f"no figure text for {lang!r}; known: {', '.join(LANGUAGES)}")
    return WORDS[lang]

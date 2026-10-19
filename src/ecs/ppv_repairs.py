"""Five repairs of the positive predictive value a receiving hospital is told.

Every repair produces a threshold and an announced PPV; each is judged against
the PPV observed on target patients it did not see.  The target ``v`` of a
transfer is the source's own PPV at the source's threshold, and it is computed
from the source alone (:func:`frozen_target`) before any target record is read.

``prior``         no target label.  The source threshold is kept; the announced
                  PPV is Bayes' rule on the source sensitivity and specificity
                  at the prevalence Saerens' EM estimates from the target's
                  unlabelled probabilities (``ecs.ppv_gap.label_free``).
``recalibrated``  n target labels.  The source threshold is kept; the announced
                  PPV is the mean probability of the flagged after a logistic
                  recalibration fitted on the n labels (``ecs.repairs``).
``resampled``     n target labels, cut in two halves by patient.  The threshold
                  is refitted on the first half for 90% sensitivity, by the same
                  conformal rank as at the source; the announced PPV is the
                  share of ill among the second half's flagged, beside its
                  one-sided Clopper-Pearson lower bound.
``certified``     n target labels.  Thresholds from a grid fixed by the source
                  scores are tested in a fixed sequence, from the highest to the
                  lowest, for H0: PPV <= v with an exact binomial test at level
                  delta; the first acceptance stops the walk, and the certified
                  threshold is the lowest one rejected before it (Learn-then-Test,
                  Angelopoulos et al. 2021).  The announced PPV is v, as a bound.
``selection``     no target label.  Weighted conformal selection (Jin & Candes,
                  arXiv:2307.09291, Algorithm 1 with deterministic pruning) at
                  FDR level q = 1 - v, with the clipped score and covariate-shift
                  weights estimated from the model's score, source against target.

The labelled draws are taken by patient, so no evaluation patient shares a
patient with a labelled record.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import beta, binom
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

from .calibration import recalibrated, sensitivity_specificity
from .conformal import conformal_quantile
from .ppv_gap import label_free
from .repairs import local_recalibration, net_benefit_of
from .transfer import ALPHA

__all__ = [
    "BUDGETS",
    "DELTA",
    "DRAWS",
    "GRID_SHARES",
    "NAMED_THRESHOLD",
    "Target",
    "certified_cut",
    "clopper_pearson_lower",
    "covariate_weights",
    "flags",
    "frozen_target",
    "labelled_draw",
    "prior_repair",
    "selection_repair",
    "sensitivity_cut",
    "transfer_cell",
    "weighted_conformal_selection",
]

Array = NDArray[np.float64]
IntArray = NDArray[np.int_]
BoolArray = NDArray[np.bool_]

BUDGETS = (25, 50, 100, 200, 400)
DRAWS = 200
DELTA = 0.10
NAMED_THRESHOLD = 0.10
# The certification grid: the source score at which this share of the source is
# flagged, from 5% to 95% in steps of 2.5 points, walked from the top.
GRID_SHARES: tuple[float, ...] = tuple(round(0.05 + 0.025 * k, 3) for k in range(37))
WEIGHT_CLIP_QUANTILE = 0.99
LOGIT_FLOOR = 1e-6


@dataclass(frozen=True)
class Target:
    """What the source alone fixes for a transfer: its threshold, its PPV, the grid.

    Thresholds are cuts on the LAC score 1 - p: a record is flagged when its
    score is at most the cut, as in ``ecs.transfer.conformal_sets``.
    """

    cut: float
    ppv: float
    sensitivity: float
    specificity: float
    grid: tuple[float, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "cut": self.cut,
            "ppv": self.ppv,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
        }


def flags(p: Array, cut: float) -> BoolArray:
    out: BoolArray = (1.0 - np.asarray(p, dtype=np.float64)) <= cut
    return out


def frozen_target(p_source: Array, y_source: IntArray) -> Target:
    """The source's 90%-sensitivity cut, its PPV there, and the certification grid."""
    p_source = np.asarray(p_source, dtype=np.float64)
    y_source = np.asarray(y_source, dtype=int)
    cut = conformal_quantile(1.0 - p_source[y_source == 1], ALPHA)
    flagged = flags(p_source, cut)
    sens, spec = sensitivity_specificity(y_source, flagged)
    grid = tuple(float(np.quantile(1.0 - p_source, s)) for s in GRID_SHARES)
    return Target(cut, float(y_source[flagged].mean()), sens, spec, grid)


def clopper_pearson_lower(ill: int, flagged: int, delta: float = DELTA) -> float:
    """One-sided exact lower bound at level 1 - delta on a binomial proportion."""
    if ill == 0:
        return 0.0
    return float(beta.ppf(delta, ill, flagged - ill + 1))


def labelled_draw(codes: IntArray, n: int, rng: np.random.Generator) -> tuple[IntArray, IntArray]:
    """n labelled records drawn by patient, and the records of every untouched patient.

    Patients are put in a random order and their records taken in that order
    until n are held; the last patient touched may give fewer than all their
    records, and the rest of that patient goes to neither side.
    """
    n_patients = int(codes.max()) + 1
    rank = np.empty(n_patients, dtype=int)
    rank[rng.permutation(n_patients)] = np.arange(n_patients)
    record_rank = rank[codes]
    order = np.argsort(record_rank, kind="stable")
    labelled = order[:n]
    last = record_rank[labelled].max()
    evaluation = np.flatnonzero(record_rank > last)
    return labelled, evaluation


def _halves(codes: IntArray, labelled: IntArray) -> tuple[IntArray, IntArray]:
    """The labelled records cut in two by patient, alternating in patient code order."""
    patients = np.unique(codes[labelled])
    first = np.isin(codes[labelled], patients[::2])
    return labelled[first], labelled[~first]


def sensitivity_cut(p: Array, y: IntArray) -> float | None:
    """The 90%-sensitivity cut by the source's conformal rank; None without an ill record."""
    ill = 1.0 - np.asarray(p, dtype=np.float64)[np.asarray(y) == 1]
    return conformal_quantile(ill, ALPHA) if ill.size else None


def certified_cut(p: Array, y: IntArray, target: Target, delta: float = DELTA) -> float | None:
    """The lowest grid threshold certified PPV > v by fixed-sequence exact tests, or None.

    The walk goes from the highest threshold (smallest cut) down and stops at the
    first null it cannot reject; a threshold flagging no labelled record has
    p-value 1 and stops it.
    """
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    certified = None
    for cut in sorted(target.grid):
        flagged = flags(p, cut)
        k, x = int(flagged.sum()), int(y[flagged].sum())
        # P(X >= x) under Binomial(k, v): the exact p-value of H0 PPV <= v.
        if k == 0 or binom.sf(x - 1, k, target.ppv) > delta:
            break
        certified = cut
    return certified


def _observed(y: IntArray, flagged: BoolArray) -> dict[str, Any]:
    ill = int(y.sum())
    k = int(flagged.sum())
    return {
        "ppv": float(y[flagged].mean()) if k else None,
        "sensitivity": float(y[flagged].sum() / ill) if ill else None,
        "net_benefit": net_benefit_of(y, flagged, NAMED_THRESHOLD),
    }


def prior_repair(
    p_source: Array, y_source: IntArray, p: Array, y: IntArray, target: Target
) -> dict[str, Any]:
    """The label-free prevalence correction, announced and observed on the whole target."""
    flagged = flags(p, target.cut)
    free = label_free(p_source, y_source, p, flagged, target.sensitivity, target.specificity)
    seen = _observed(y, flagged)
    announced = free["ppv_recipe_estimated_prevalence"]
    error = (
        abs(announced - seen["ppv"]) if announced is not None and seen["ppv"] is not None else None
    )
    return {
        "announced": announced,
        "observed": seen["ppv"],
        "abs_error": error,
        "prevalence_estimated": free["prevalence_estimated"],
        "sensitivity": seen["sensitivity"],
        "net_benefit": seen["net_benefit"],
    }


def covariate_weights(p_source: Array, p: Array) -> tuple[Array, Array, float]:
    """dP_target / dP_source as a function of the score, for the source and target records.

    A logistic classifier on a cubic spline of logit(p) separates target from
    source; its odds times n_source / n_target is the density ratio.  Weights
    are clipped at the 99th centile of the pooled weights.  Also returns the
    effective sample size of the source weights.
    """

    def logit(q: Array) -> Array:
        q = np.clip(np.asarray(q, dtype=np.float64), LOGIT_FLOOR, 1.0 - LOGIT_FLOOR)
        out: Array = np.log(q / (1.0 - q))
        return out

    x = np.concatenate([logit(p_source), logit(p)])[:, None]
    side = np.concatenate([np.zeros(len(p_source)), np.ones(len(p))])
    model = make_pipeline(SplineTransformer(n_knots=5, degree=3), LogisticRegression(C=1.0))
    model.fit(x, side)
    prob = np.clip(model.predict_proba(x)[:, 1], LOGIT_FLOOR, 1.0 - LOGIT_FLOOR)
    w = prob / (1.0 - prob) * len(p_source) / len(p)
    w = np.minimum(w, np.quantile(w, WEIGHT_CLIP_QUANTILE))
    w_source, w_target = w[: len(p_source)], w[len(p_source) :]
    ess = float(w_source.sum() ** 2 / (w_source**2).sum())
    return w_source, w_target, ess


def weighted_conformal_selection(
    p_cal: Array, y_cal: IntArray, w_cal: Array, p: Array, w: Array, q: float
) -> BoolArray:
    """Jin & Candes' WCS with the clipped score, selecting Y = 1 at FDR level q.

    With V(x, y) = M 1{y > 0} - p(x) and threshold c = 0, a calibration record
    counts below test record l only when it is healthy and scored above it, so
    the weighted conformal p-value is
        p_l = (A_l + w_l) / (W + w_l),  A_l = sum of w_i over healthy i with p_i > p_l,
    W the sum of all calibration weights.  For each j the auxiliary p-values
    p_l^(j) = (A_l + w_j 1{p_j > p_l}) / (W + w_j), with p_j^(j) = 0, run
    through BH give |R_j|; j passes the first step when p_j <= q |R_j| / m,
    and deterministic pruning keeps the passing j with |R_j| <= r*, the
    largest r such that at least r passing j have |R_j| <= r.
    """
    p_cal = np.asarray(p_cal, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    m = len(p)
    healthy = np.asarray(y_cal) == 0
    order_h = np.argsort(p_cal[healthy])
    sorted_h = p_cal[healthy][order_h]
    # tail[k] = sum of healthy weights at sorted positions >= k.
    tail = np.concatenate([np.cumsum(w_cal[healthy][order_h][::-1])[::-1], [0.0]])
    above = tail[np.searchsorted(sorted_h, p, side="right")]
    total = float(w_cal.sum())
    pvalue = (above + w) / (total + w)

    order = np.argsort(-p, kind="stable")  # ascending p-values
    a_sorted, p_sorted = above[order], p[order]
    position = np.arange(1, m + 1)
    size = np.empty(m, dtype=int)
    chunk = max(1, 4_000_000 // max(m, 1))
    for start in range(0, m, chunk):
        js = np.arange(start, min(start + chunk, m))  # sorted positions of j
        wj = w[order][js][:, None]
        pj = p_sorted[js][:, None]
        aux = (a_sorted[None, :] + wj * (pj > p_sorted[None, :])) / (total + wj)
        # Entries ranked before j move down one place when j's zero goes first.
        shifted = position[None, :] + (position[None, :] < (js[:, None] + 1))
        passing = aux <= q * shifted / m
        passing[np.arange(len(js)), js] = False
        last = np.where(passing, shifted, 0).max(axis=1)
        size[order[js]] = np.maximum(last, 1)
    first = pvalue <= q * size / m
    out: BoolArray = first & (size <= _largest_r(np.sort(size[first])))
    return out


def _largest_r(sizes: IntArray) -> int:
    """max r in 1..len(sizes) with #{sizes <= r} >= r, or 0."""
    counts = np.searchsorted(sizes, np.arange(1, sizes.size + 1), side="right")
    ok = np.flatnonzero(counts >= np.arange(1, sizes.size + 1))
    return int(ok[-1] + 1) if ok.size else 0


def selection_repair(
    p_source: Array, y_source: IntArray, p: Array, y: IntArray, target: Target
) -> dict[str, Any]:
    """WCS at q = 1 - v on the whole target; the false discovery proportion it reached."""
    w_source, w_target, ess = covariate_weights(p_source, p)
    q = 1.0 - target.ppv
    selected = weighted_conformal_selection(p_source, y_source, w_source, p, w_target, q)
    seen = _observed(y, selected)
    return {
        "q": q,
        "n_selected": int(selected.sum()),
        "nonempty": bool(selected.any()),
        "fdp": 1.0 - seen["ppv"] if seen["ppv"] is not None else 0.0,
        "sensitivity": seen["sensitivity"],
        "net_benefit": seen["net_benefit"],
        "source_effective_size": ess,
    }


def _quartiles(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    return {"n": len(values), "median": float(med), "q1": float(q1), "q3": float(q3)}


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _budget(
    p: Array, y: IntArray, codes: IntArray, target: Target, n: int, rng: np.random.Generator
) -> dict[str, Any]:
    """Methods 2 to 4 over DRAWS labelled draws of n target records."""
    errors: dict[str, list[float]] = {"recalibrated": [], "resampled": []}
    sens: dict[str, list[float]] = {"recalibrated": [], "resampled": [], "certified": []}
    nb: dict[str, list[float]] = {"recalibrated": [], "resampled": [], "certified": []}
    resampled_defined = bound_held = certified_found = violated = clear_violation = 0
    for _ in range(DRAWS):
        lab, ev = labelled_draw(codes, n, rng)
        p_ev, y_ev = p[ev], y[ev]
        # 2. Recalibration on the n labels, source threshold kept.
        a, b, _kind = local_recalibration(p[lab], y[lab])
        kept = flags(p_ev, target.cut)
        seen = _observed(y_ev, kept)
        if seen["ppv"] is not None:
            announced = float(recalibrated(p_ev[kept], a, b).mean())
            errors["recalibrated"].append(abs(announced - seen["ppv"]))
        _push(sens["recalibrated"], nb["recalibrated"], seen)
        # 3. Threshold refitted on one half, PPV read on the other.
        first, second = _halves(codes, lab)
        cut = sensitivity_cut(p[first], y[first])
        if cut is not None:
            held = flags(p[second], cut)
            k = int(held.sum())
            seen = _observed(y_ev, flags(p_ev, cut))
            if k and seen["ppv"] is not None:
                resampled_defined += 1
                announced = float(y[second][held].mean())
                errors["resampled"].append(abs(announced - seen["ppv"]))
                bound_held += seen["ppv"] >= clopper_pearson_lower(int(y[second][held].sum()), k)
            _push(sens["resampled"], nb["resampled"], seen)
        # 4. Certified threshold on all n labels.
        cut = certified_cut(p[lab], y[lab], target)
        if cut is not None:
            certified_found += 1
            seen = _observed(y_ev, flags(p_ev, cut))
            k_ev = int(flags(p_ev, cut).sum())
            if seen["ppv"] is not None and seen["ppv"] < target.ppv:
                violated += 1
                ill = int(round(seen["ppv"] * k_ev))
                upper = 1.0 - clopper_pearson_lower(k_ev - ill, k_ev, 0.025)
                clear_violation += upper < target.ppv
            _push(sens["certified"], nb["certified"], seen)
    return {
        "n": n,
        "recalibrated": {
            "abs_error": _quartiles(errors["recalibrated"]),
            "sensitivity": _mean(sens["recalibrated"]),
            "net_benefit": _mean(nb["recalibrated"]),
        },
        "resampled": {
            "share_defined": resampled_defined / DRAWS,
            "abs_error": _quartiles(errors["resampled"]),
            "share_bound_held": bound_held / resampled_defined if resampled_defined else None,
            "sensitivity": _mean(sens["resampled"]),
            "net_benefit": _mean(nb["resampled"]),
        },
        "certified": {
            "share_threshold_exists": certified_found / DRAWS,
            "violation_rate": violated / DRAWS,
            "violation_rate_given_threshold": violated / certified_found
            if certified_found
            else None,
            "clear_violation_rate": clear_violation / DRAWS,
            "sensitivity": _mean(sens["certified"]),
            "net_benefit": _mean(nb["certified"]),
        },
        "_errors": errors,
        "_violations": violated,
        "_found": certified_found,
    }


def _push(sens: list[float], nb: list[float], seen: dict[str, Any]) -> None:
    if seen["sensitivity"] is not None:
        sens.append(seen["sensitivity"])
    nb.append(seen["net_benefit"])


def transfer_cell(
    p_source: Array,
    y_source: IntArray,
    p: Array,
    y: IntArray,
    patients: pd.Series,
    target: Target,
    seed: int,
    budgets: tuple[int, ...] = BUDGETS,
) -> dict[str, Any]:
    """Every repair on one transfer, against the target ``target`` fixed beforehand."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    codes = pd.factorize(patients.to_numpy())[0].astype(int)
    rng = np.random.default_rng(seed)
    rungs = [_budget(p, y, codes, target, n, rng) for n in budgets if n < math.ceil(len(p) / 2)]
    return {
        "prior": prior_repair(p_source, y_source, p, y, target),
        "selection": selection_repair(p_source, y_source, p, y, target),
        "budgets": rungs,
    }

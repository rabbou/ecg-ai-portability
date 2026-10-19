"""Study P-A: what a label-free reweighting repairs, and how each model's probability holds.

A2, weighted coverage.  If P(Y|X) is the same for inpatients and outpatients,
the density ratio between the two populations at a fixed diagnosis is the
covariate ratio w(x) up to a constant:

    p_t(x|y) / p_s(x|y) = w(x) * pi_s(y) / pi_t(y),

so the per-label (Mondrian) conformal threshold can be weighted with w(x) alone,
and the constant cancels from the weighted quantile (Tibshirani et al.,
arXiv:1904.06019, applied within each class).  w(x) is the odds of a domain
classifier that tells calibration inpatients from unlabelled target records,
cross-fitted so no record is weighted by a model that saw it, and capped at the
99th centile of the calibration weights.

The falsifier is fixed in the protocol before any result was read: on the
composite, the 95% Wilson interval of the weighted sensitivity lies inside
[85%, 95%]; on each judged anomaly (at least 50 target cases), the sensitivity
clears 80% in a one-sided score test at 0.025 after Holm's correction.  Both
are computed at the effective size min(target cases, Kish size of the
calibration cases' weights), because the threshold rests on the weighted
calibration cases and an interval at the full target count would overstate
what the weights support.

A3, selective prediction.  The decision is the study's: flag when p reaches the
calibration threshold that keeps 90% of inpatient cases.  Confidence is the
distance |logit p - logit t| from that threshold; abstaining on the least
confident records traces the risk-coverage curve, summarised by its
generalised area (AUGRC, Traub et al., arXiv:2407.01032), and by the
sensitivity left on the 80% of records kept.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .calibration import _logit, brier, slope_intercept
from .conformal import lac_scores, lac_scores_all, weighted_conformal_quantile
from .metrics import effective_sample_size, wilson_interval

__all__ = [
    "ANOMALY_FLOOR",
    "COMPOSITE_BAND",
    "JUDGED_MIN_CASES",
    "DomainWeights",
    "a2_verdict",
    "augrc",
    "calibration_cell",
    "domain_weights",
    "effective_cases",
    "holm_rejections",
    "integrated_calibration_index",
    "selective_sensitivity",
    "superiority_p",
    "weighted_perlabel_sets",
]

Array = NDArray[np.float64]
IntArray = NDArray[np.int_]
BoolArray = NDArray[np.bool_]

DOMAIN_FOLDS = 5
WEIGHT_CAP_QUANTILE = 0.99
PROBABILITY_EPS = 1e-6
JUDGED_MIN_CASES = 50
COMPOSITE_BAND = (0.85, 0.95)
ANOMALY_FLOOR = 0.80
ONE_SIDED_LEVEL = 0.025
RETAINED = 0.80
BOOTSTRAP_DRAWS = 500
ICI_GRID = 200


@dataclass(frozen=True)
class DomainWeights:
    """w(x) for each calibration and each target record, and how separable the two were."""

    calibration: Array
    target: Array
    domain_auroc: float
    cap: float


def domain_weights(x_cal: Array, x_tgt: Array, seed: int = 0) -> DomainWeights:
    """Cross-fitted odds p(target|x) / p(source|x), rescaled by n_cal / n_tgt and capped.

    The classifier is an L2 logistic regression on standardised covariates, its
    penalty chosen by an inner five-fold search on log loss.  The rescaling turns
    the classifier's odds into the density ratio p_t(x) / p_s(x); the cap at
    the 99th centile of the calibration weights applies to both sides, so a
    target record never weighs more than the heaviest calibration record kept.
    """
    x_cal, x_tgt = np.asarray(x_cal, dtype=np.float64), np.asarray(x_tgt, dtype=np.float64)
    x = np.vstack([x_cal, x_tgt])
    domain = np.r_[np.zeros(len(x_cal), dtype=int), np.ones(len(x_tgt), dtype=int)]
    model = make_pipeline(
        StandardScaler(),
        GridSearchCV(
            LogisticRegression(max_iter=5000),
            {"C": np.logspace(-4, 4, 10)},
            cv=DOMAIN_FOLDS,
            scoring="neg_log_loss",
        ),
    )
    folds = StratifiedKFold(DOMAIN_FOLDS, shuffle=True, random_state=seed)
    p = cross_val_predict(model, x, domain, cv=folds, method="predict_proba")[:, 1]
    p = np.clip(p, PROBABILITY_EPS, 1 - PROBABILITY_EPS)
    odds = p / (1 - p) * len(x_cal) / len(x_tgt)
    cap = float(np.quantile(odds[: len(x_cal)], WEIGHT_CAP_QUANTILE))
    weights = np.minimum(odds, cap)
    return DomainWeights(
        calibration=weights[: len(x_cal)],
        target=weights[len(x_cal) :],
        domain_auroc=float(roc_auc_score(domain, p)),
        cap=cap,
    )


def weighted_perlabel_sets(
    p_cal: Array, y_cal: IntArray, w_cal: Array, p_tgt: Array, w_tgt: Array, alpha: float
) -> BoolArray:
    """The (m, 2) per-label conformal sets with each class's quantile weighted by w(x).

    Class c's threshold for target record j is the Tibshirani quantile of the
    calibration records of class c, each at its w, with record j's own w at +inf.
    """
    y_cal = np.asarray(y_cal, dtype=int)
    cal = lac_scores(np.column_stack([1 - p_cal, p_cal]), y_cal)
    tgt = lac_scores_all(np.column_stack([1 - p_tgt, p_tgt]))
    thresholds = np.column_stack(
        [
            weighted_conformal_quantile(cal[y_cal == c], w_cal[y_cal == c], w_tgt, alpha)
            if (y_cal == c).any()
            else np.full(len(p_tgt), math.inf)
            for c in (0, 1)
        ]
    )
    inside: BoolArray = tgt <= thresholds
    return inside


def effective_cases(n_target_cases: int, case_weights: Array) -> int:
    """The size a weighted sensitivity is judged at: the smaller of the two samples it rests on."""
    if n_target_cases == 0 or len(case_weights) == 0:
        return 0
    return int(min(n_target_cases, math.floor(effective_sample_size(case_weights))))


def superiority_p(sensitivity: float, n: int, floor: float = ANOMALY_FLOOR) -> float:
    """One-sided score-test p-value of H0: sensitivity <= floor, at sample size n."""
    if n <= 0:
        return 1.0
    z = (sensitivity - floor) / math.sqrt(floor * (1 - floor) / n)
    return float(stats.norm.sf(z))


def holm_rejections(p_values: dict[str, float], level: float = ONE_SIDED_LEVEL) -> dict[str, bool]:
    """Holm's step-down: the hypotheses rejected at family-wise ``level``."""
    order = sorted(p_values, key=lambda k: p_values[k])
    rejected = {k: False for k in p_values}
    for i, key in enumerate(order):
        if p_values[key] > level / (len(order) - i):
            break
        rejected[key] = True
    return rejected


def a2_verdict(composite: dict[str, Any], anomalies: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The protocol's A2 decision from weighted-sensitivity rows.

    Each row holds ``coverage_pos`` and ``n_effective``.  The composite passes
    when its 95% Wilson interval at the effective size lies in COMPOSITE_BAND;
    an anomaly passes when Holm rejects sensitivity <= 80%; the repair holds
    when the composite and a strict majority of judged anomalies pass.
    """
    n = composite["n_effective"]
    low, high = wilson_interval(round(composite["coverage_pos"] * n), n) if n else (0.0, 1.0)
    composite_pass = COMPOSITE_BAND[0] <= low and high <= COMPOSITE_BAND[1]
    p_values = {
        label: superiority_p(row["coverage_pos"], row["n_effective"])
        for label, row in anomalies.items()
    }
    passed = holm_rejections(p_values)
    majority = sum(passed.values()) > len(passed) / 2
    return {
        "composite_interval": [low, high],
        "composite_pass": composite_pass,
        "anomaly_p_values": p_values,
        "anomaly_pass": passed,
        "anomalies_passed": sum(passed.values()),
        "anomalies_judged": len(passed),
        "repair_holds": bool(composite_pass and majority),
    }


def integrated_calibration_index(y: IntArray, p: Array) -> float:
    """Mean |p - smoothed observed rate at p| over the records.

    Austin & Steyerberg (Stat Med 2019) smooth the outcome on the prediction
    with loess; here the smoother is a Gaussian kernel on logit(p) with
    Silverman's bandwidth, evaluated on a grid of ``ICI_GRID`` points and
    interpolated, which keeps a bootstrap of thousands of records linear in n.
    """
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    z = _logit(p)
    spread = float(np.std(z))
    if spread == 0.0:
        return float(abs(p.mean() - y.mean()))
    bandwidth = 1.06 * spread * len(z) ** -0.2
    grid = np.linspace(z.min(), z.max(), ICI_GRID)
    kernel = np.exp(-0.5 * ((grid[:, None] - z[None, :]) / bandwidth) ** 2)
    observed = np.interp(z, grid, kernel @ y / kernel.sum(axis=1))
    return float(np.mean(np.abs(p - observed)))


def _confidence_order(p: Array, threshold: float) -> IntArray:
    distance = np.abs(_logit(np.asarray(p, dtype=np.float64)) - _logit(np.array([threshold]))[0])
    order: IntArray = np.argsort(-distance, kind="stable")
    return order


def augrc(y: IntArray, p: Array, threshold: float) -> float:
    """Area under the generalised risk-coverage curve of the flag p >= threshold.

    Records are kept most confident first; at each coverage i/n the generalised
    risk is the count of wrong decisions among the i kept, over n.  The area is
    the mean of that risk over the n coverages.
    """
    y = np.asarray(y, dtype=int)
    wrong = ((np.asarray(p) >= threshold).astype(int) != y)[_confidence_order(p, threshold)]
    return float(np.mean(np.cumsum(wrong) / len(y)))


def selective_sensitivity(
    y: IntArray, p: Array, threshold: float, retained: float = RETAINED
) -> tuple[float | None, float]:
    """Sensitivity on the most confident ``retained`` share, and the share of cases kept."""
    y = np.asarray(y, dtype=int)
    if y.sum() == 0:
        return None, 0.0
    kept = _confidence_order(p, threshold)[: math.ceil(retained * len(y))]
    cases = kept[y[kept] == 1]
    share_kept = len(cases) / int(y.sum())
    if len(cases) == 0:
        return None, share_kept
    return float(np.mean(np.asarray(p)[cases] >= threshold)), share_kept


def _cell(y: IntArray, p: Array, threshold: float) -> dict[str, float | None]:
    slope: float | None = None
    intercept: float | None = None
    if 0 < y.sum() < len(y):
        # Separable outcomes leave the slope undefined; the draw count reports it.
        with contextlib.suppress(ValueError):
            slope, intercept = slope_intercept(y, p)
    sensitivity, cases_kept = selective_sensitivity(y, p, threshold)
    return {
        "slope": slope,
        "intercept": intercept,
        "ici": integrated_calibration_index(y, p),
        "brier": brier(y, p),
        "augrc": augrc(y, p, threshold),
        "sensitivity_at_80": sensitivity,
        "cases_kept_at_80": cases_kept,
    }


def calibration_cell(
    y: IntArray, p: Array, threshold: float, draws: int = BOOTSTRAP_DRAWS, seed: int = 0
) -> dict[str, Any]:
    """Each A3 metric with its 95% percentile interval over record bootstrap draws.

    The threshold is the calibration cohort's and stays fixed across draws: the
    interval is about the target sample, not about where the threshold fell.
    A draw where a metric is undefined (no case drawn, separable outcomes) is
    left out of that metric's interval, and the count kept is reported.
    """
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=np.float64)
    point = _cell(y, p, threshold)
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {k: [] for k in point}
    for _ in range(draws):
        idx = rng.integers(0, len(y), len(y))
        for key, value in _cell(y[idx], p[idx], threshold).items():
            if value is not None:
                samples[key].append(value)
    out: dict[str, Any] = {"n": len(y), "n_pos": int(y.sum()), "threshold": threshold}
    for key, value in point.items():
        kept = samples[key]
        out[key] = value
        out[f"{key}_low"] = float(np.percentile(kept, 2.5)) if kept else None
        out[f"{key}_high"] = float(np.percentile(kept, 97.5)) if kept else None
        out[f"{key}_draws"] = len(kept)
    return out

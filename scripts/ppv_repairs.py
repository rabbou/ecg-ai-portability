"""Which repair makes the announced PPV equal the observed one, and for how many labels.

Three hospitals carry the conclusion: Columbia (EchoNext, four models, from
inpatients to outpatients and the emergency department), Beth Israel (MIMIC-IV-ECG
as a target of the seven-source rotation, machine-read labels) and Chongqing
(the PTB-XL infarction model).  Shandong, the other rotation targets, the CODE
sets and ZZU are a secondary reserve; the pairs whose target is the source's
own population are the in-distribution controls.

The run has two passes.  The first fixes every transfer's target v, the source's
PPV at the source's threshold, from the source scores alone, and records the
digest of that table.  The second reads the target records and applies the five
repairs of ``ecs.ppv_repairs``: methods 1 and 5 on the whole target with no
label, methods 2 to 4 over 200 draws of n target labels, n = 25 to 400.

Writes ``results/ppv_repairs.json``, with the summaries, the falsifiers and the rows
of the main hospitals and the controls, and ``results/ppv_repairs_reserve.json``
with the secondary rows.  Needs EchoNext with its four arms' stored scores, and the
patient tables of PTB-XL, Shandong and Chongqing.

Usage: .venv/bin/python scripts/ppv_repairs.py
"""

from __future__ import annotations

import hashlib
import json
import time
import zlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from score_pairs import Pair, echonext, echonext_available, infarction, rotation_seven, rounded
from sklearn.metrics import roc_auc_score

from ecs.config import RESULTS_DIR
from ecs.ppv_gap import label_free
from ecs.ppv_repairs import (
    BUDGETS,
    DELTA,
    DRAWS,
    NAMED_THRESHOLD,
    Target,
    flags,
    frozen_target,
    transfer_cell,
)
from ecs.provenance import provenance_block
from ecs.repairs import patient_bootstrap_weights

OUT = RESULTS_DIR / "ppv_repairs.json"
# The secondary targets' rows, apart so that each file stays under the 512 kB bar.
RESERVE = RESULTS_DIR / "ppv_repairs_reserve.json"
MAIN = ("columbia", "beth_israel", "chongqing")
HOSPITAL_OF_TARGET = {
    "outpatient": "columbia",
    "emergency": "columbia",
    "mimic": "beth_israel",
    "acs": "chongqing",
    "sph": "shandong",
}
# A pair is run when both sides hold this many ill records, and summarised when
# the target holds MIN_ILL_SUMMARY: below it the observed PPV of a draw's
# evaluation set rests on a handful of patients.
MIN_ILL_RUN = 10
MIN_ILL_SUMMARY = 20
BOOTSTRAP = 200
CLOSE = 0.02  # the falsifier's two points
VIOLATION_BAR = 0.15  # delta plus 0.05 for 200 dependent draws
FALSIFIER_BUDGET = 100
DIGITS = 3  # per-transfer figures; the summaries keep six

PRODUCERS = [
    "scripts/ppv_repairs.py",
    "scripts/score_pairs.py",
    "src/ecs/ppv_repairs.py",
    "src/ecs/ppv_gap.py",
    "src/ecs/repairs.py",
    "src/ecs/calibration.py",
    "src/ecs/conformal.py",
    "src/ecs/transfer.py",
]


def key_of(pair: Pair) -> str:
    return "/".join((pair.family, pair.model, pair.source, pair.target, pair.label))


def hospital_of(pair: Pair) -> str:
    """The receiving hospital, or 'control' when the target is the source's own population."""
    if pair.in_distribution:
        return "control"
    return HOSPITAL_OF_TARGET.get(pair.target, pair.target)


def runnable(pair: Pair) -> bool:
    return bool(pair.y_source.sum() >= MIN_ILL_RUN and pair.y_target.sum() >= MIN_ILL_RUN)


def freeze_targets(pairs: Iterable[Pair]) -> dict[str, Target]:
    """Pass one: every target v from the source arrays only."""
    return {key_of(p): frozen_target(p.p_source, p.y_source) for p in pairs}


def digest(targets: dict[str, Target]) -> str:
    table = {k: t.as_dict() for k, t in sorted(targets.items())}
    return hashlib.sha256(json.dumps(table, sort_keys=True).encode()).hexdigest()


def seed_of(key: str) -> int:
    """A seed fixed by the transfer's name, not by its place in the run."""
    return zlib.crc32(key.encode())


def prior_bootstrap(pair: Pair, target: Target, seed: int) -> list[float | None]:
    """Method 1's absolute error on BOOTSTRAP redraws of the target's patients."""
    assert pair.target_patients is not None
    weights = patient_bootstrap_weights(
        pair.target_patients.reset_index(drop=True), BOOTSTRAP, seed
    )
    out: list[float | None] = []
    for w in weights:
        idx = np.repeat(np.arange(len(w)), w)
        p, y = pair.p_target[idx], pair.y_target[idx]
        flagged = flags(p, target.cut)
        free = label_free(
            pair.p_source, pair.y_source, p, flagged, target.sensitivity, target.specificity
        )
        recipe = free["ppv_recipe_estimated_prevalence"]
        out.append(
            abs(recipe - y[flagged].mean()) if recipe is not None and flagged.any() else None
        )
    return out


def measure(pair: Pair, target: Target, commit: str) -> dict[str, Any]:
    assert pair.target_patients is not None
    key = key_of(pair)
    seed = seed_of(key)
    cell = transfer_cell(
        pair.p_source,
        pair.y_source,
        pair.p_target,
        pair.y_target,
        pair.target_patients.reset_index(drop=True),
        target,
        seed,
    )
    row = {
        "hospital": hospital_of(pair),
        "family": pair.family,
        "model": pair.model,
        "source": pair.source,
        "target": pair.target,
        "label": pair.label,
        "n_target": len(pair.y_target),
        "n_target_ill": int(pair.y_target.sum()),
        "auroc_target": float(roc_auc_score(pair.y_target, pair.p_target)),
        "summarised": bool(pair.y_target.sum() >= MIN_ILL_SUMMARY),
        "v": target.ppv,
        "source_cut": target.cut,
        "seed": seed,
        "commit": commit,
        **cell,
    }
    if row["hospital"] in MAIN:
        row["_prior_bootstrap"] = prior_bootstrap(pair, target, seed + 1)
    return row


def _quartiles(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    return {"n": len(values), "median": float(med), "q1": float(q1), "q3": float(q3)}


def _mean(values: Iterable[float | None]) -> float | None:
    kept = [v for v in values if v is not None]
    return float(np.mean(kept)) if kept else None


def median_error_upper(rows: list[dict[str, Any]], seed: int) -> float | None:
    """97.5th centile of the median of method 1's absolute error, transfers and patients redrawn."""
    table = np.array(
        [[np.nan if e is None else e for e in r["_prior_bootstrap"]] for r in rows], dtype=float
    )
    if table.size == 0:
        return None
    rng = np.random.default_rng(seed)
    medians = [
        np.nanmedian(table[rng.integers(len(rows), size=len(rows)), b]) for b in range(BOOTSTRAP)
    ]
    return float(np.nanpercentile(medians, 97.5))


def budget_summary(rows: list[dict[str, Any]], n: int) -> dict[str, Any]:
    """Methods 2 to 4 at budget n, every draw of every transfer pooled."""
    rungs = [b for r in rows for b in r["budgets"] if b["n"] == n]
    if not rungs:
        return {"n": n, "transfers": 0}
    draws = DRAWS * len(rungs)
    found = sum(b["_found"] for b in rungs)
    violations = sum(b["_violations"] for b in rungs)
    return {
        "n": n,
        "transfers": len(rungs),
        "recalibrated_abs_error": _quartiles(
            [e for b in rungs for e in b["_errors"]["recalibrated"]]
        ),
        "resampled_abs_error": _quartiles([e for b in rungs for e in b["_errors"]["resampled"]]),
        "resampled_share_defined": _mean(b["resampled"]["share_defined"] for b in rungs),
        "resampled_share_bound_held": _mean(b["resampled"]["share_bound_held"] for b in rungs),
        "certified_share_threshold_exists": found / draws,
        "certified_violation_rate": violations / draws,
        "certified_violation_rate_given_threshold": violations / found if found else None,
        "certified_clear_violation_rate": _mean(
            b["certified"]["clear_violation_rate"] for b in rungs
        ),
        "sensitivity": {
            m: _mean(b[m]["sensitivity"] for b in rungs)
            for m in ("recalibrated", "resampled", "certified")
        },
        "net_benefit": {
            m: _mean(b[m]["net_benefit"] for b in rungs)
            for m in ("recalibrated", "resampled", "certified")
        },
    }


def hospital_summary(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    kept = [r for r in rows if r["summarised"]]
    prior = [r["prior"]["abs_error"] for r in kept if r["prior"]["abs_error"] is not None]
    out: dict[str, Any] = {
        "transfers": len(kept),
        "transfers_not_summarised": len(rows) - len(kept),
        "prior": {
            "abs_error": _quartiles(prior),
            "sensitivity": _mean(r["prior"]["sensitivity"] for r in kept),
            "net_benefit": _mean(r["prior"]["net_benefit"] for r in kept),
        },
        "selection": {
            "mean_fdp": _mean(r["selection"]["fdp"] for r in kept),
            "mean_q": _mean(r["selection"]["q"] for r in kept),
            "share_fdp_above_q": _mean(
                float(r["selection"]["fdp"] > r["selection"]["q"]) for r in kept
            ),
            "share_nonempty": _mean(float(r["selection"]["nonempty"]) for r in kept),
            "sensitivity": _mean(r["selection"]["sensitivity"] for r in kept),
            "net_benefit": _mean(r["selection"]["net_benefit"] for r in kept),
        },
        "budgets": [budget_summary(kept, n) for n in BUDGETS],
    }
    if kept and "_prior_bootstrap" in kept[0]:
        out["prior"]["median_abs_error_upper_97_5"] = median_error_upper(kept, seed)
    return out


def falsifiers(summary: dict[str, Any]) -> dict[str, Any]:
    """The two falsifiers of protocol section 7, decided from the summary."""
    uppers = {h: summary[h]["prior"].get("median_abs_error_upper_97_5") for h in MAIN[1:]}
    label_free_enough = all(u is not None and u < CLOSE for u in uppers.values())
    at_n = {
        h: next(b for b in summary[h]["budgets"] if b["n"] == FALSIFIER_BUDGET)
        for h in MAIN
        if summary[h]["transfers"]
    }
    return {
        "label_free_outside_columbia": {
            "rule": "the 97.5th bootstrap centile of method 1's median absolute error is under "
            "2 points at both Beth Israel and Chongqing",
            "upper_bounds": uppers,
            "fired": label_free_enough,
        },
        "certification_at_100": {
            "rule": f"method 4's violation rate exceeds {VIOLATION_BAR} in a hospital, or a "
            "threshold exists in under half of the draws",
            "violation_rate": {h: b["certified_violation_rate"] for h, b in at_n.items()},
            "share_threshold_exists": {
                h: b["certified_share_threshold_exists"] for h, b in at_n.items()
            },
            "fired_violation": any(
                b["certified_violation_rate"] > VIOLATION_BAR for b in at_n.values()
            ),
            "fired_existence": any(
                b["certified_share_threshold_exists"] < 0.5 for b in at_n.values()
            ),
        },
    }


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    """The row without the per-draw lists the summaries pool, each error as its median."""
    out = {k: v for k, v in row.items() if not k.startswith("_")}
    out["budgets"] = []
    for rung in row["budgets"]:
        kept = {k: v for k, v in rung.items() if not k.startswith("_")}
        for method in ("recalibrated", "resampled"):
            block = dict(kept[method])
            block["abs_error"] = block["abs_error"].get("median")
            kept[method] = block
        out["budgets"].append(kept)
    return out


def write(path: Path, head: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """One transfer per line, so a rerun's diff names the transfers that moved."""
    lines = ",\n".join("  " + json.dumps(rounded(r, DIGITS), separators=(",", ":")) for r in rows)
    text = json.dumps(rounded(head), indent=1)[:-2] + ',\n "rows": [\n' + lines + "\n ]\n}\n"
    path.write_text(text)


def published_control() -> float:
    """The in-distribution median gap the article prints, read from its file."""
    ppv = json.loads((RESULTS_DIR / "echonext_ppv_gap.json").read_text())
    return float(ppv["summary"]["echonext"]["in_distribution"]["median_abs_gap_points"])


def main() -> None:
    started = time.time()
    if not echonext_available():
        raise SystemExit("EchoNext and its stored scores are needed: Columbia is a main hospital")
    pairs = [
        p
        for p in (
            *echonext(patients=True),
            *infarction(patients=True),
            *rotation_seven(patients=True),
        )
        if runnable(p)
    ]
    targets = freeze_targets(pairs)
    frozen = digest(targets)
    commit = provenance_block(PRODUCERS)["commit"]
    rows = [measure(p, targets[key_of(p)], commit) for p in pairs]
    groups = sorted({r["hospital"] for r in rows})
    summary = {
        h: hospital_summary([r for r in rows if r["hospital"] == h], seed_of(h)) for h in groups
    }
    result = {
        "question": (
            "Which repair makes the PPV announced to a receiving hospital equal the PPV it "
            "observes, and for how many local labels?"
        ),
        "protocol": "lab research/pieces/protocoles_etudes_2026-10-16.md, section 7",
        "target_v": "the source's PPV at the source's 90%-sensitivity threshold, fixed in a "
        "first pass from the source scores alone",
        "targets_digest": frozen,
        "budgets": list(BUDGETS),
        "draws": DRAWS,
        "delta": DELTA,
        "net_benefit_threshold": NAMED_THRESHOLD,
        "main_hospitals": list(MAIN),
        "published_control_median_gap": published_control(),
        "summary": summary,
        "falsifiers": falsifiers(summary),
        "seconds": round(time.time() - started, 1),
        "provenance": provenance_block(PRODUCERS),
    }
    kept = [public_row(r) for r in rows if r["hospital"] in (*MAIN, "control")]
    reserve = [public_row(r) for r in rows if r["hospital"] not in (*MAIN, "control")]
    write(OUT, result | {"reserve_rows": RESERVE.name}, kept)
    write(
        RESERVE,
        {
            "rows_of": OUT.name,
            "targets_digest": frozen,
            "commit": commit,
            "provenance": result["provenance"],
        },
        reserve,
    )
    print(f"{len(rows)} transfers in {time.time() - started:.1f} s")


if __name__ == "__main__":
    main()

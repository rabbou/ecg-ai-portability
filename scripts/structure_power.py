"""How often the A2 composite condition is met by a recalibration exactly on target.

The condition asks the whole 95% Wilson interval of composite sensitivity, at the size
the verdict uses, to lie in COMPOSITE_BAND: a floor and a ceiling.  For each target and
size reading of ``results/structure.json``, this script counts the detected cases k for
which ``a2_verdict`` would pass the composite, and sums the binomial probability of those
k at the target sensitivity.  It reads only committed files and runs in a second, so the
probabilities live in their own file, ``results/structure_power.json``, rather than in
structure.json, whose regeneration needs EchoNext.
"""

from __future__ import annotations

import json
from math import comb
from typing import Any

from ecs.config import RESULTS_DIR
from ecs.metrics import wilson_interval
from ecs.provenance import provenance_block
from ecs.structure import COMPOSITE_BAND

PRODUCERS = ["scripts/structure_power.py", "src/ecs/metrics.py", "src/ecs/structure.py"]


def passing_cases(n: int) -> list[int]:
    """The detected-case counts k whose Wilson interval at n lies in COMPOSITE_BAND."""
    low, high = COMPOSITE_BAND
    return [
        k
        for k in range(n + 1)
        if low <= wilson_interval(k, n)[0] and wilson_interval(k, n)[1] <= high
    ]


def pass_probability(n: int, sensitivity: float) -> float:
    """P(the composite condition is met) when k ~ Binomial(n, sensitivity)."""
    return sum(comb(n, k) * sensitivity**k * (1 - sensitivity) ** (n - k) for k in passing_cases(n))


def power(summary: dict[str, Any]) -> dict[str, Any]:
    """For each target and reading, the size, the passing range and the probability."""
    sensitivity = 1 - summary["alpha"]
    out: dict[str, Any] = {}
    for context in summary["verdict_contexts"]:
        out[context] = {}
        for reading in summary["readings"]:
            sizes = {
                arms[summary["verdict_covariates"]]["n"]
                for arms in summary["a2"][context][reading].values()
            }
            # The weights are shared by every arm, so every arm is judged at one size.
            (n,) = sizes
            k = passing_cases(n)
            out[context][reading] = {
                "n": n,
                "passing_cases": [k[0], k[-1]],
                "passing_low": k[0] / n,
                "passing_high": k[-1] / n,
                "probability": pass_probability(n, sensitivity),
            }
    return out


def main() -> None:
    summary = json.loads((RESULTS_DIR / "structure.json").read_text())
    result = {
        "sensitivity": 1 - summary["alpha"],
        "band": list(COMPOSITE_BAND),
        "power": power(summary),
        "provenance": provenance_block(PRODUCERS),
    }
    (RESULTS_DIR / "structure_power.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

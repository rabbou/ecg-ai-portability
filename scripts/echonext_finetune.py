"""Fine-tune ECGFounder end to end on EchoNext's train split, one run per seed (study P-A, A1).

The published 12-lead Net1D starts from its own weights; its 150-way dense head is
replaced by twelve sigmoid outputs, one per EchoNext label, and every parameter
trains.  The input is the frozen arm's: the stored (12, 2500) z-scored tracing at
250 Hz brought to 500 Hz by ``resample_poly(2, 1)``, so the frozen and the
fine-tuned arm differ by training alone.

The epoch is chosen on the same tenth of training patients the study's ResNet1d
held out (``patient_split`` with seed 0, whatever the training seed), never on
the inpatient calibration cohort, which the threshold is set on afterwards.

Weights, per-epoch checkpoints and scores are derived restricted data and stay
under ``$ECS_ECHONEXT_DERIVED``: ``models/ecgfounder_ft_s<seed>.pt`` with its
sha256 in ``scores/ecgfounder_ft_s<seed>.json``, and the probabilities with the
1024-d features of every validation and test record in the matching ``.npz``.
A run cut short resumes at the epoch after its last checkpoint.

Usage (overnight, under the Mac's GPU lock):
    PYTORCH_ENABLE_MPS_FALLBACK=1 /usr/bin/lockf ~/.claude/locks/mac-gpu.lock \
        .venv/bin/python scripts/echonext_finetune.py --seeds 0 1 2
    .venv/bin/python scripts/echonext_finetune.py --trial 40   # time 40 steps, write nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from echonext_scores import (
    HOLDOUT_SHARE,
    SCORES_DIR,
    Tracings,
    device,
    labels_of,
    mean_auroc,
    peak_memory,
    scored_positions,
)
from numpy.typing import NDArray
from scipy.signal import resample_poly

from ecs.echonext import DERIVED_DIR, LABELS, UNIT, read_metadata, transfer_cohorts
from ecs.encoders import allowlisted_load, ecgfounder_net, machine_info, verified
from ecs.splits import patient_split

EPOCHS = 10
BATCH = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
HOLDOUT_SEED = 0
SEEDS = (0, 1, 2)
MODELS_DIR = DERIVED_DIR / "models"
RUN_LOG = DERIVED_DIR / "ecgfounder-ft-run.log"

Float32Array = NDArray[np.float32]
Reader = Callable[[NDArray[np.int_]], Float32Array]


def arm_name(seed: int) -> str:
    return f"ecgfounder_ft_s{seed}"


def at_500hz(read: Reader) -> Reader:
    """The reader, with each tracing brought from 250 to 500 Hz as the frozen arm does."""
    return lambda positions: resample_poly(read(positions), 2, 1, axis=-1).astype(np.float32)


def finetune_net(seed: int) -> torch.nn.Module:
    """ECGFounder on its published weights, the dense head replaced by twelve outputs.

    The head is drawn under ``seed``, which is all the seed changes in the network.
    """
    state = allowlisted_load(verified("ecgfounder/12_lead_ECGFounder.pth"))["state_dict"]
    net = ecgfounder_net()
    net.load_state_dict(state, strict=True)
    torch.manual_seed(seed)
    net.dense = torch.nn.Linear(net.dense.in_features, len(LABELS))
    return net


@dataclass(frozen=True)
class Plan:
    """What one training run reads: the positions, their labels, and where it files itself."""

    fit: NDArray[np.int_]
    holdout: NDArray[np.int_]
    y_fit: Float32Array
    y_hold: Float32Array
    seed: int
    checkpoint: Path
    epochs: int = EPOCHS
    batch: int = BATCH


def batches(
    read: Reader, fit: NDArray[np.int_], order: NDArray[np.int_], size: int
) -> Iterator[tuple[NDArray[np.int_], Float32Array]]:
    """(rows of ``order``, tracings) per batch, the next batch read while this one trains."""
    starts = range(0, len(order), size)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = None
        for start in starts:
            rows = order[start : start + size]
            future = pool.submit(read, fit[rows])
            if pending is not None:
                yield pending[0], pending[1].result()
            pending = (rows, future)
        if pending is not None:
            yield pending[0], pending[1].result()


def predict(
    net: torch.nn.Module, read: Reader, positions: NDArray[np.int_], dev: torch.device
) -> tuple[Float32Array, Float32Array]:
    """Sigmoid probabilities and the features before the head, for ``positions``."""
    net.eval()
    probs, feats = [], []
    order = np.arange(len(positions))
    with torch.no_grad():
        for _, x in batches(read, positions, order, 128):
            logits, features = net(torch.from_numpy(x).to(dev))
            probs.append(torch.sigmoid(logits).float().cpu().numpy())
            feats.append(features.float().cpu().numpy())
    return np.concatenate(probs), np.concatenate(feats)


def cosine_factor(step: int, total: int) -> float:
    """The learning-rate multiplier: one at step 0, falling on a half cosine to zero at total."""
    return 0.5 * (1.0 + math.cos(math.pi * min(step, total) / total))


def train(
    net: torch.nn.Module, read: Reader, plan: Plan, dev: torch.device, log: Callable[[str], None]
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    """Train for ``plan.epochs``, resuming from ``plan.checkpoint``; the best state and history.

    The checkpoint is written after every epoch with the optimiser and scheduler,
    so a resumed run takes the same steps as one never interrupted.
    """
    net.to(dev)
    optimiser = torch.optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    steps = plan.epochs * math.ceil(len(plan.fit) / plan.batch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda s: cosine_factor(s, steps))
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    if plan.checkpoint.exists():
        saved = torch.load(plan.checkpoint, map_location="cpu", weights_only=True)
        net.load_state_dict(saved["model"])
        optimiser.load_state_dict(saved["optimiser"])
        scheduler.load_state_dict(saved["scheduler"])
        best_state = saved["best"]
        history = json.loads(plan.checkpoint.with_suffix(".json").read_text())
    for epoch in range(len(history), plan.epochs):
        started = time.time()
        net.train()
        order = np.random.default_rng([plan.seed, epoch]).permutation(len(plan.fit))
        losses = []
        for rows, x in batches(read, plan.fit, order, plan.batch):
            y = torch.from_numpy(plan.y_fit[rows]).to(dev)
            optimiser.zero_grad()
            logits = net(torch.from_numpy(x).to(dev))[0]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            optimiser.step()
            scheduler.step()
            losses.append(float(loss))
        score = mean_auroc(plan.y_hold, predict(net, read, plan.holdout, dev)[0])
        history.append(
            {
                "seed": plan.seed,
                "epoch": epoch + 1,
                "loss": float(np.mean(losses)),
                "holdout_mean_auroc": score,
                "seconds": round(time.time() - started, 1),
            }
        )
        log(json.dumps(history[-1]))
        if score >= max(h["holdout_mean_auroc"] for h in history):
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        plan.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": net.state_dict(),
                "optimiser": optimiser.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best": best_state,
            },
            plan.checkpoint,
        )
        plan.checkpoint.with_suffix(".json").write_text(json.dumps(history, indent=2) + "\n")
    if best_state is None:
        raise RuntimeError(f"seed {plan.seed}: no epoch was trained")
    return best_state, history


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def plan_for(meta: pd.DataFrame, seed: int) -> Plan:
    train_rows = transfer_cohorts(meta).training
    part = patient_split(
        meta["patient_key"].iloc[train_rows],
        {"fit": 1 - HOLDOUT_SHARE, "holdout": HOLDOUT_SHARE},
        HOLDOUT_SEED,
    ).to_numpy()
    fit, holdout = train_rows[part == "fit"], train_rows[part == "holdout"]
    return Plan(
        fit=fit,
        holdout=holdout,
        y_fit=labels_of(meta, fit),
        y_hold=labels_of(meta, holdout),
        seed=seed,
        checkpoint=MODELS_DIR / f"{arm_name(seed)}-training.pt",
    )


def log_line(text: str) -> None:
    print(text, flush=True)
    with RUN_LOG.open("a") as handle:
        handle.write(text + "\n")


def run_seed(meta: pd.DataFrame, read: Reader, seed: int) -> None:
    """Train one seed, file its best weights, score validation and test, write its record."""
    started = time.time()
    plan = plan_for(meta, seed)
    dev = device()
    net = finetune_net(seed)
    best_state, history = train(net, read, plan, dev, log_line)
    net.load_state_dict(best_state)
    weights = MODELS_DIR / f"{arm_name(seed)}.pt"
    torch.save(best_state, weights)
    positions = scored_positions(meta)
    probs, features = predict(net, read, positions, dev)
    info = {
        "arm": arm_name(seed),
        "description": "ECGFounder 12-lead Net1D fine-tuned end to end, 12 sigmoid outputs",
        "weights_start": "huggingface.co/PKUDigitalHealth/ECGFounder 12_lead_ECGFounder.pth (MIT)",
        "weights_file": str(weights),
        "weights_sha256": sha256(weights),
        "input": (
            f"(12, 5000): the stored (12, 2500) at 250 Hz, unit {UNIT}, resampled to 500 Hz "
            "by scipy.signal.resample_poly(2, 1)"
        ),
        "features": "1024-d mean over time before the dense head",
        "fit_records": len(plan.fit),
        "holdout_records": len(plan.holdout),
        "epochs": plan.epochs,
        "batch": plan.batch,
        "optimiser": f"AdamW(lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY}), cosine to 0",
        "seed": seed,
        "holdout_seed": HOLDOUT_SEED,
        "history": history,
        "chosen_epoch": int(np.argmax([h["holdout_mean_auroc"] for h in history]) + 1),
        "seconds_this_invocation": round(time.time() - started, 1),
        "device": str(dev),
        **peak_memory(),
        "machine": machine_info(),
    }
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    keys = meta["ecg_key"].to_numpy()[positions]
    np.savez(
        SCORES_DIR / f"{arm_name(seed)}.npz",
        ecg_key=keys,
        probs=probs,
        features=features,
        labels=np.array(LABELS),
    )
    (SCORES_DIR / f"{arm_name(seed)}.json").write_text(json.dumps(info, indent=2) + "\n")
    log_line(json.dumps({"seed": seed, "done": True, "weights_sha256": info["weights_sha256"]}))


def trial(meta: pd.DataFrame, read: Reader, steps: int) -> None:
    """Time ``steps`` training steps and project one seed's ten epochs; writes nothing."""
    plan = plan_for(meta, 0)
    dev = device()
    net = finetune_net(0).to(dev).train()
    optimiser = torch.optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    order = np.random.default_rng(0).permutation(len(plan.fit))[: (steps + 2) * plan.batch]
    started = 0.0
    for done, (rows, x) in enumerate(batches(read, plan.fit, order, plan.batch)):
        if done == 2:  # the first steps compile kernels; they are not timed
            if dev.type == "mps":
                torch.mps.synchronize()
            started = time.time()
        y = torch.from_numpy(plan.y_fit[rows]).to(dev)
        optimiser.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            net(torch.from_numpy(x).to(dev))[0], y
        )
        loss.backward()
        optimiser.step()
        float(loss)
    rate = steps * plan.batch / (time.time() - started)
    print(
        json.dumps(
            {
                "trial_steps": steps,
                "ecg_per_second": round(rate, 1),
                "fit_records": len(plan.fit),
                "hours_per_seed_training": round(plan.epochs * len(plan.fit) / rate / 3600, 2),
                **peak_memory(),
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--trial", type=int, help="time this many training steps, write nothing")
    args = parser.parse_args()
    meta = read_metadata()
    read = at_500hz(Tracings(meta))
    if args.trial:
        trial(meta, read, args.trial)
        return
    for seed in args.seeds:
        if (SCORES_DIR / f"{arm_name(seed)}.json").exists():
            log_line(json.dumps({"seed": seed, "skipped": "scores already filed"}))
            continue
        run_seed(meta, read, seed)


if __name__ == "__main__":
    main()

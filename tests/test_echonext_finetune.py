"""The A1 fine-tuning loop of scripts/echonext_finetune.py, on a toy network and toy tracings.

The real run is ten hours on the Mac's GPU per three seeds; these tests hold what
would silently spoil it: a resumed run that drifts from the uninterrupted one, a
prefetch that loses or reorders a batch, and a schedule that does not reach zero.
The first test pays torch's one-time lazy import of its optimiser machinery,
about 8 s on the Mac; the training itself takes 0.1 s.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from echonext_finetune import Plan, batches, cosine_factor, train


class Toy(torch.nn.Module):
    """(logits, features) like ECGFounder's Net1D with return_features, three outputs."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.body = torch.nn.Linear(8, 4)
        self.dense = torch.nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.relu(self.body(x.mean(dim=1)))
        return self.dense(features), features


def read(positions: np.ndarray) -> np.ndarray:
    """A tracing per position, the same on every call: (n, 2 leads, 8 samples)."""
    rows = [np.random.default_rng(int(i)).normal(size=(2, 8)) for i in positions]
    return np.asarray(rows, dtype=np.float32)


def plan(tmp_path: Path) -> Plan:
    rng = np.random.default_rng(7)
    y = (rng.random((60, 3)) < 0.4).astype(np.float32)
    return Plan(
        fit=np.arange(40),
        holdout=np.arange(40, 60),
        y_fit=y[:40],
        y_hold=y[40:],
        seed=1,
        checkpoint=tmp_path / "toy-training.pt",
        epochs=3,
        batch=8,
    )


def test_a_run_cut_short_resumes_onto_the_same_weights(tmp_path: Path) -> None:
    """A crash after epoch 1 then a restart must file the weights an uninterrupted run
    files, or the seed's model in ~/data/echonext-derived/models is not that seed's."""
    whole_dir, cut_dir = tmp_path / "whole", tmp_path / "cut"
    whole_state, whole_history = train(Toy(), read, plan(whole_dir), torch.device("cpu"), print)

    lines: list[str] = []

    def crash_on_epoch_two(line: str) -> None:
        lines.append(line)
        if len(lines) == 2:
            raise KeyboardInterrupt  # killed before epoch 2's checkpoint is written

    with pytest.raises(KeyboardInterrupt):
        train(Toy(), read, plan(cut_dir), torch.device("cpu"), crash_on_epoch_two)
    cut_state, cut_history = train(Toy(), read, plan(cut_dir), torch.device("cpu"), print)

    assert [h["epoch"] for h in cut_history] == [1, 2, 3]
    assert [h["loss"] for h in cut_history] == pytest.approx([h["loss"] for h in whole_history])
    for key, tensor in whole_state.items():
        torch.testing.assert_close(cut_state[key], tensor)


def test_prefetch_yields_every_batch_once_in_order() -> None:
    """The prefetched batches must be the order's rows, each once, with their tracings."""
    fit = np.arange(100, 110)
    order = np.array([3, 1, 4, 0, 9, 2, 6])
    seen = list(batches(read, fit, order, 3))
    assert [rows.tolist() for rows, _ in seen] == [[3, 1, 4], [0, 9, 2], [6]]
    np.testing.assert_array_equal(seen[2][1], read(np.array([106])))


def test_the_schedule_starts_at_one_and_ends_at_zero() -> None:
    """The learning rate is the full rate at step 0, half at mid-run and zero at the end."""
    assert cosine_factor(0, 100) == 1.0
    assert cosine_factor(50, 100) == pytest.approx(0.5)
    assert cosine_factor(100, 100) == pytest.approx(0.0)
    assert cosine_factor(130, 100) == pytest.approx(0.0)

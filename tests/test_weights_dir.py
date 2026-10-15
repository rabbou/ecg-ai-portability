"""The published weights are found from any checkout of the clone, worktrees included."""

from __future__ import annotations

from pathlib import Path

import pytest

from ecs import encoders


class TestTheWeightsDirectory:
    """A linked worktree reads the main checkout's untracked weights."""

    def test_a_linked_worktree_reads_the_main_checkout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ECS_WEIGHTS_DIR", raising=False)
        main, linked = tmp_path / "main", tmp_path / "linked"
        (main / ".git/worktrees/linked").mkdir(parents=True)
        (main / "data/weights").mkdir(parents=True)
        linked.mkdir()
        (linked / ".git").write_text(f"gitdir: {main / '.git/worktrees/linked'}\n")
        assert encoders.weights_dir(linked) == main / "data/weights"
        assert encoders.weights_dir(main) == main / "data/weights"

    def test_a_checkout_with_its_own_weights_or_an_override_keeps_them(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ECS_WEIGHTS_DIR", raising=False)
        (tmp_path / "data/weights").mkdir(parents=True)
        (tmp_path / ".git").write_text("gitdir: /elsewhere/.git/worktrees/x\n")
        assert encoders.weights_dir(tmp_path) == tmp_path / "data/weights"
        monkeypatch.setenv("ECS_WEIGHTS_DIR", "/some/where")
        assert encoders.weights_dir(tmp_path) == Path("/some/where")

    def test_a_relative_gitdir_is_read_from_the_worktree(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """git worktree.useRelativePaths writes the gitdir relative to the worktree."""
        monkeypatch.delenv("ECS_WEIGHTS_DIR", raising=False)
        main = tmp_path / "main"
        linked = main / ".claude/worktrees/linked"
        (main / ".git/worktrees/linked").mkdir(parents=True)
        (main / "data/weights").mkdir(parents=True)
        linked.mkdir(parents=True)
        (linked / ".git").write_text("gitdir: ../../../.git/worktrees/linked\n")
        assert encoders.weights_dir(linked) == main / "data/weights"


class TestPeakMemory:
    """getrusage counts bytes on macOS and KiB on Linux; the record is bytes on both."""

    @pytest.mark.parametrize(("platform", "factor"), [("darwin", 1), ("linux", 1024)])
    def test_the_peak_is_in_bytes(
        self, platform: str, factor: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class Usage:
            ru_maxrss = 2_000

        monkeypatch.setattr(encoders.sys, "platform", platform)
        monkeypatch.setattr(encoders.resource, "getrusage", lambda _: Usage())
        assert encoders.peak_rss_bytes() == 2_000 * factor

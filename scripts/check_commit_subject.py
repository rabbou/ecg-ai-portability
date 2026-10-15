"""Refuse a commit subject over 72 characters or one that opens on a framing colon.

Run by pre-commit at the commit-msg stage with the message file as its argument.
A subject is the first line that is not a comment.  Git's own subjects for a
merge, a revert, a fixup or a squash are left to git.

Usage: .venv/bin/python scripts/check_commit_subject.py .git/COMMIT_EDITMSG
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

MAX_LENGTH = 72
GIT_WRITTEN = ("Merge ", "Revert ", "fixup! ", "squash! ", "amend! ")
# "Study: rebuild the results", "fix(ci): ..." -- a label and a colon before the sentence.
FRAMING_COLON = re.compile(r"^[^\s:]+(?:\s+[^\s:]+)?:\s")


def subject_of(message: str) -> str:
    for line in message.splitlines():
        if line.strip() and not line.startswith("#"):
            return line.rstrip()
    return ""


def problems(subject: str) -> list[str]:
    if subject.startswith(GIT_WRITTEN):
        return []
    found = []
    if len(subject) > MAX_LENGTH:
        found.append(f"the subject is {len(subject)} characters; the limit is {MAX_LENGTH}")
    if FRAMING_COLON.match(subject):
        found.append("the subject opens on a label and a colon; start with the verb")
    return found


def main(argv: list[str]) -> int:
    subject = subject_of(Path(argv[1]).read_text())
    found = problems(subject)
    for problem in found:
        print(f"commit-msg: {problem}: {subject!r}", file=sys.stderr)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

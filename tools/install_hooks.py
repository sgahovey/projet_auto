#!/usr/bin/env python3
"""Install project git hooks."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path


def set_executable(path: Path) -> None:
    """Ensure POSIX execute bits are set when possible."""
    try:
        mode = path.stat().st_mode
        path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except (FileNotFoundError, PermissionError, OSError):
        # On some platforms (e.g. Windows), chmod may fail. Ignore silently.
        pass


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    hooks_dir = repo_root / ".githooks"
    if not hooks_dir.exists():
        print("Missing .githooks directory; cannot install hooks.", file=sys.stderr)
        return 1

    post_commit = hooks_dir / "post-commit"
    if post_commit.exists():
        set_executable(post_commit)

    try:
        subprocess.run(
            ["git", "config", "core.hooksPath", ".githooks"],
            cwd=repo_root,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f"Failed to configure hooks path: {exc.stderr}", file=sys.stderr)
        return exc.returncode

    print("core.hooksPath configured to .githooks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Assistant interactif pour créer des commits Conventional Commits."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

COMMIT_TYPES = [
    "feat",
    "fix",
    "ci",
    "test",
    "docs",
    "refactor",
    "chore",
    "deploy",
]

SCOPES = [
    "init",
    "backend",
    "frontend",
    "docker",
    "ansible",
    "monitoring",
    "github-actions",
    "prod",
]


def run_git_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def ensure_git_repository() -> None:
    result = run_git_command(["rev-parse", "--is-inside-work-tree"])
    if result.returncode != 0 or result.stdout.strip() != "true":
        sys.stderr.write("Ce dossier n'est pas un dépôt Git initialisé.\n")
        sys.exit(1)


def ensure_staged_changes() -> None:
    result = run_git_command(["diff", "--cached", "--name-only"])
    if result.returncode != 0:
        sys.stderr.write("Impossible de vérifier les fichiers indexés.\n")
        sys.exit(1)
    if not result.stdout.strip():
        sys.stderr.write("Aucun fichier dans l'index. Lancez `git add` avant de committer.\n")
        sys.exit(1)


def prompt_choice(title: str, options: list[str], allow_skip: bool = False) -> str | None:
    print(title)
    for idx, option in enumerate(options, start=1):
        print(f"{idx}) {option}")
    if allow_skip:
        print("0) Aucun")

    while True:
        raw = input("> ").strip()
        if allow_skip and raw == "0":
            return None
        if raw.isdigit():
            index = int(raw)
            if 1 <= index <= len(options):
                return options[index - 1]
        print("Choix invalide, réessayez.")


def prompt_message() -> str:
    while True:
        message = input("Message du commit : ").strip()
        if message:
            return message
        print("Le message ne peut pas être vide.")


def main() -> int:
    ensure_git_repository()
    ensure_staged_changes()

    commit_type = prompt_choice("Type de commit :", COMMIT_TYPES)
    scope = prompt_choice("Scope (facultatif) :", SCOPES, allow_skip=True)

    difficulty_raw = prompt_choice(
        "Niveau de difficulté (1 très simple, 5 très complexe) :", ["1", "2", "3", "4", "5"]
    )
    difficulty_label = f"Diff-{difficulty_raw}"

    message = prompt_message()

    if scope:
        commit_header = f"{commit_type}({scope}) [{difficulty_label}]: {message}"
    else:
        commit_header = f"{commit_type} [{difficulty_label}]: {message}"

    result = run_git_command(["commit", "-m", commit_header])
    if result.returncode != 0:
        sys.stderr.write(result.stderr or "Échec du commit.\n")
        return result.returncode

    print(result.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
